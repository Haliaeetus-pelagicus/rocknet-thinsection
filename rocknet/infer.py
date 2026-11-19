"""Load the round-8 checkpoint and run the official identification protocol."""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from .decode import (
    apply_informal_collapse,
    apply_mineral_prior,
    apply_pair_rerank,
    cosine_class_logits,
    decode_group_and_lith,
)
from .model import RockNet

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE_ROOT / "assets"
WEIGHTS = PACKAGE_ROOT / "weights"

IMAGE_SIZE = 384
TTA_VIEWS = 8
PAIR_GAP = 0.32
PRIOR_WEIGHT = 0.4
VOTE_MIN = 5
MINERAL_THRESHOLD = 0.45

_RUNTIME = None


def _checkpoint_bytes() -> bytes:
    parts = sorted(WEIGHTS.glob("model.pt.part*"))
    if not parts:
        raise FileNotFoundError(f"No checkpoint parts in {WEIGHTS}")
    blob = bytearray()
    for part in parts:
        blob.extend(part.read_bytes())
    digest_path = WEIGHTS / "SHA256.txt"
    if digest_path.exists():
        import hashlib

        expected = digest_path.read_text(encoding="utf-8").split()[0].strip()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError("Joined checkpoint does not match SHA256.txt")
    return bytes(blob)


def load_checkpoint(device: torch.device) -> dict:
    single = WEIGHTS / "model.pt"
    if single.exists():
        return torch.load(single, map_location=device, weights_only=False)
    buffer = io.BytesIO(_checkpoint_bytes())
    return torch.load(buffer, map_location=device, weights_only=False)


def _transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _tta_views(tensor: torch.Tensor, n_views: int) -> list[torch.Tensor]:
    views = [tensor]
    if n_views >= 2:
        views.append(torch.flip(tensor, dims=[-1]))
    if n_views >= 3:
        views.append(torch.flip(tensor, dims=[-2]))
    if n_views >= 4:
        views.append(torch.rot90(tensor, 1, dims=[-2, -1]))
    if n_views >= 5:
        views.append(torch.rot90(tensor, 2, dims=[-2, -1]))
    if n_views >= 6:
        views.append(torch.rot90(tensor, 3, dims=[-2, -1]))
    if n_views >= 7:
        views.append(torch.flip(torch.rot90(tensor, 1, dims=[-2, -1]), dims=[-1]))
    if n_views >= 8:
        views.append(torch.flip(torch.rot90(tensor, 1, dims=[-2, -1]), dims=[-2]))
    return views[: max(1, n_views)]


def get_runtime(device: str | None = None) -> dict:
    global _RUNTIME
    if _RUNTIME is not None and device is None:
        return _RUNTIME
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = load_checkpoint(torch_device)
    state = checkpoint["model"]
    n_pairs = int(checkpoint.get("n_pairs") or 0)
    if "pair_head.weight" in state:
        n_pairs = int(state["pair_head.weight"].shape[0])
    model = RockNet(checkpoint["n_lithology"], checkpoint["n_minerals"], checkpoint["n_group"], n_pairs=n_pairs)
    model.load_state_dict(state, strict=False)
    model.to(torch_device)
    model.eval()
    maps = json.loads((ASSETS / "label_maps.json").read_text(encoding="utf-8"))
    tables = torch.load(ASSETS / "decode_tables.pt", map_location="cpu", weights_only=False)
    pair_index = [tuple(pair) for pair in checkpoint.get("pair_index") or []]
    runtime = {
        "model": model,
        "maps": maps,
        "device": torch_device,
        "mask": tables["group_lith_mask"],
        "prior": tables["mineral_prior"],
        "collapse": [tuple(pair) for pair in tables["informal_collapse"]],
        "pair_index": pair_index,
        "arcface_s": float(checkpoint.get("arcface_s", 16.0)),
        "use_arcface": bool(checkpoint.get("use_arcface", True)),
        "epoch": int(checkpoint.get("epoch") or 0),
    }
    if device is None:
        _RUNTIME = runtime
    return runtime


def _photo_key(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    if len(parts) == 3 and parts[1] in {"1", "2", "3"} and parts[2] in {"1", "2", "3"}:
        return parts[0]
    return stem


def _decode_one(runtime: dict, image: Image.Image, topk: int, mineral_thr: float, tta_views: int, pair_gap: float, prior_weight: float) -> dict:
    model = runtime["model"]
    device = runtime["device"]
    tensor = _transforms()(image.convert("RGB")).unsqueeze(0).to(device)
    liths, groups, minerals, pairs = [], [], [], []
    base = None
    with torch.no_grad():
        for index, view in enumerate(_tta_views(tensor, tta_views)):
            output = model(view)
            if index == 0:
                base = output
            if runtime["use_arcface"]:
                liths.append(cosine_class_logits(output["feat"], model.lithology_head.weight, runtime["arcface_s"]))
            else:
                liths.append(output["lithology"])
            groups.append(output["group"])
            minerals.append(output["minerals"])
            if "pair" in output:
                pairs.append(output["pair"])
        lith_logits = torch.stack(liths).mean(0)
        group_logits = torch.stack(groups).mean(0)
        mineral_logits = torch.stack(minerals).mean(0)
        pair_logits = torch.stack(pairs).mean(0) if pairs else None
        lith_prob, group_prob = decode_group_and_lith(group_logits, lith_logits, runtime["mask"].to(device))
        lith_prob = apply_mineral_prior(lith_prob, mineral_logits, runtime["prior"], prior_weight)
        lith_prob = apply_pair_rerank(lith_prob, pair_logits, runtime["pair_index"], max_gap=pair_gap)
        lith_prob = apply_informal_collapse(lith_prob, runtime["collapse"])
        lith_prob = lith_prob[0]
        group_prob = group_prob[0]
        mineral_prob = mineral_logits.sigmoid()[0]
        seg_fraction = base["seg"][0].softmax(dim=0).mean(dim=(1, 2)).cpu()
    names = runtime["maps"]["lithology_classes"]
    group_names = runtime["maps"]["group_classes"]
    mineral_names = runtime["maps"]["mineral_classes"]
    top_values, top_index = torch.topk(lith_prob, k=min(topk, lith_prob.numel()))
    contents = []
    for class_index, fraction in enumerate(seg_fraction.tolist()):
        if fraction < 0.01:
            continue
        name = "other/matrix" if class_index == 0 else mineral_names[class_index - 1]
        contents.append({"name": name, "fraction": float(fraction)})
    contents.sort(key=lambda item: item["fraction"], reverse=True)
    minerals = [
        {"name": mineral_names[index], "probability": float(value)}
        for index, value in enumerate(mineral_prob.tolist())
        if value >= mineral_thr
    ]
    minerals.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "group": {
            "name": group_names[int(group_prob.argmax())],
            "probability": float(group_prob.max()),
        },
        "lithology": [
            {"name": names[int(index)], "probability": float(value)}
            for value, index in zip(top_values.tolist(), top_index.tolist())
        ],
        "minerals": minerals,
        "area_fraction": contents,
    }


def predict_images(
    paths: list[str | Path],
    topk: int = 5,
    mineral_threshold: float = MINERAL_THRESHOLD,
    tta_views: int = TTA_VIEWS,
    pair_gap: float = PAIR_GAP,
    prior_weight: float = PRIOR_WEIGHT,
    device: str | None = None,
) -> dict:
    """Identify one or more tiles. Nine tiles from one field are majority-voted when at least five agree."""
    runtime = get_runtime(device)
    tiles = []
    for path in paths:
        image_path = Path(path)
        result = _decode_one(runtime, Image.open(image_path), topk, mineral_threshold, tta_views, pair_gap, prior_weight)
        result["file"] = image_path.name
        result["view"] = _photo_key(image_path.name)
        tiles.append(result)
    buckets: dict[str, list[dict]] = {}
    for tile in tiles:
        buckets.setdefault(tile["view"], []).append(tile)
    votes = []
    for view, members in buckets.items():
        counts = Counter(member["lithology"][0]["name"] for member in members if member["lithology"])
        if not counts:
            continue
        winner, n_votes = counts.most_common(1)[0]
        applied = len(members) >= 3 and n_votes >= VOTE_MIN
        votes.append({"view": view, "tiles": len(members), "winner": winner, "votes": n_votes, "applied": applied})
        if applied:
            for member in members:
                member["voted_lithology"] = winner
    return {
        "checkpoint_epoch": runtime["epoch"],
        "image_size": IMAGE_SIZE,
        "tta_views": tta_views,
        "pair_rerank_gap": pair_gap,
        "mineral_threshold": mineral_threshold,
        "tiles": tiles,
        "photo_vote": votes,
    }


def predict_image(path: str | Path, **kwargs) -> dict:
    report = predict_images([path], **kwargs)
    tile = report["tiles"][0]
    tile["checkpoint_epoch"] = report["checkpoint_epoch"]
    tile["protocol"] = {
        "image_size": report["image_size"],
        "tta_views": report["tta_views"],
        "pair_rerank_gap": report["pair_rerank_gap"],
        "mineral_threshold": report["mineral_threshold"],
    }
    return tile
