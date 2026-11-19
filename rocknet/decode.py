"""Official round-8 decoding: joint posterior, mineral prior, pair rerank, vernacular collapse."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_class_logits(feat: torch.Tensor, weight: torch.Tensor, scale: float = 16.0) -> torch.Tensor:
    feat = F.normalize(feat.float(), dim=1)
    weight = F.normalize(weight.float(), dim=1)
    return scale * (feat @ weight.t())


def decode_group_and_lith(
    group_logits: torch.Tensor,
    lith_logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lithology and group probabilities after the hierarchical joint posterior."""
    group_prob = group_logits.softmax(dim=1)
    lith_prob = lith_logits.softmax(dim=1)
    allowed = mask.to(device=lith_logits.device, dtype=lith_prob.dtype)
    conditional = lith_prob.unsqueeze(1) * allowed.unsqueeze(0)
    conditional = conditional / conditional.sum(dim=2, keepdim=True).clamp_min(1e-8)
    joint_lith = (group_prob.unsqueeze(2) * conditional).sum(dim=1)
    return joint_lith, group_prob


def apply_mineral_prior(
    lith_prob: torch.Tensor,
    mineral_logits: torch.Tensor,
    prior: torch.Tensor,
    weight: float = 0.4,
) -> torch.Tensor:
    """Up-weight lithology names whose teaching-table minerals match the detected minerals."""
    if weight <= 0:
        return lith_prob
    prior = prior.to(device=lith_prob.device, dtype=lith_prob.dtype)
    agreement = mineral_logits.sigmoid() @ prior.transpose(0, 1)
    agreement = agreement / prior.sum(dim=1).clamp_min(1.0)
    boosted = lith_prob * agreement.clamp_min(1e-4).pow(weight)
    return boosted / boosted.sum(dim=1, keepdim=True).clamp_min(1e-8)


def apply_pair_rerank(
    lith_prob: torch.Tensor,
    pair_logits: torch.Tensor | None,
    pair_index: list[tuple[int, int]],
    max_gap: float = 0.32,
) -> torch.Tensor:
    """Swap the top two names when they are a known confusing pair and the gap is below max_gap."""
    if pair_logits is None or not pair_index or pair_logits.numel() == 0:
        return lith_prob
    top2_values, top2_index = lith_prob.topk(2, dim=1)
    gap = top2_values[:, 0] - top2_values[:, 1]
    boosted = lith_prob.clone()
    n_pairs = min(pair_logits.size(1), len(pair_index))
    for pair_id in range(n_pairs):
        left, right = pair_index[pair_id]
        is_pair = ((top2_index[:, 0] == left) & (top2_index[:, 1] == right)) | (
            (top2_index[:, 0] == right) & (top2_index[:, 1] == left)
        )
        swap = is_pair & (gap < max_gap)
        if not bool(swap.any()):
            continue
        prefer_left = pair_logits[:, pair_id] >= 0
        winner = torch.where(
            prefer_left,
            torch.full_like(top2_index[:, 0], left),
            torch.full_like(top2_index[:, 0], right),
        )
        rows = swap.nonzero(as_tuple=False).squeeze(1)
        boosted[rows, winner[rows]] = boosted[rows, winner[rows]] + 0.12
    return boosted / boosted.sum(dim=1, keepdim=True).clamp_min(1e-8)


def apply_informal_collapse(lith_prob: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Fold a fine formal name into the vernacular class used on the teaching slides."""
    if not pairs:
        return lith_prob
    out = lith_prob.clone()
    for fine, coarse in pairs:
        if fine >= out.size(1) or coarse >= out.size(1):
            continue
        out[:, coarse] = out[:, coarse] + out[:, fine]
        out[:, fine] = 0
    return out / out.sum(dim=1, keepdim=True).clamp_min(1e-8)
