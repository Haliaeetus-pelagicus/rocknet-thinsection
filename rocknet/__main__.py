"""Command-line identification with the round-8 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .infer import predict_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify a petrographic thin-section image with the round-8 RockNet checkpoint.")
    parser.add_argument("images", nargs="+", type=Path, help="Image files or directories of tiles. A directory is read as one group of files.")
    parser.add_argument("--topk", type=int, default=5, help="Number of lithology names to return.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Mineral probability threshold.")
    parser.add_argument("--tta-views", type=int, default=8, help="Number of test-time views. Use 8 for the published protocol.")
    parser.add_argument("--pair-gap", type=float, default=0.32, help="Maximum top-2 gap for confusing-pair reranking.")
    parser.add_argument("--device", default=None, help="cuda or cpu. Default: cuda when available.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file. The same object is printed to stdout.")
    args = parser.parse_args()
    paths = []
    allowed = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    for item in args.images:
        if item.is_dir():
            paths.extend(sorted(p for p in item.iterdir() if p.is_file() and p.suffix.lower() in allowed))
        else:
            paths.append(item)
    if not paths:
        parser.error("No images found.")
    report = predict_images(
        paths,
        topk=args.topk,
        mineral_threshold=args.threshold,
        tta_views=args.tta_views,
        pair_gap=args.pair_gap,
        device=args.device,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
