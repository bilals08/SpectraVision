# Multi-positive InfoNCE with decoy repulsion.

from typing import Dict, Tuple
import math

import torch
import torch.nn.functional as F


def multi_positive_info_nce_with_decoy(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    decoys: torch.Tensor,
    temperature: float = 0.07,
    threshold: float = 1e-4,
    decoy_weight: float = 1.0,
    pos_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Compute multi-positive InfoNCE with explicit decoy negatives.
    anchors = F.normalize(anchors, dim=1)
    positives = F.normalize(positives, dim=1)
    decoys = F.normalize(decoys, dim=1)

    B = anchors.size(0)
    device = anchors.device

    sim_pos = anchors @ positives.T  # [B, P]
    sim_dec = anchors @ decoys.T     # [B, D]

    logits_pos = sim_pos / temperature  # [B, P]
    safe_decoy_weight = decoy_weight if decoy_weight > 0 else 1.0
    logits_dec = sim_dec / temperature + math.log(safe_decoy_weight)  # [B, D]
    logits = torch.cat([logits_pos, logits_dec], dim=1)  # [B, P+D]

    P = positives.size(0)
    D = decoys.size(0)


    if pos_mask is not None:
        if pos_mask.dtype != torch.bool:
            user_mask = pos_mask.to(dtype=torch.bool, device=device)
        else:
            user_mask = pos_mask.to(device=device)
        if user_mask.shape != (B, P):
            raise ValueError(f"pos_mask must have shape (B, P) = ({B}, {P}), got {tuple(user_mask.shape)}")
        pos_mask_block = user_mask
    else:

        if P == B:
            # Pair each anchor with positives that resemble its paired positive.
            pos_mask_block = (positives @ positives.T >= threshold)
            pos_mask_block = pos_mask_block.to(device=device)
            pos_mask_block.fill_diagonal_(True)
        else:

            pos_mask_block = (sim_pos >= threshold).to(device=device)

            any_pos = pos_mask_block.any(dim=1)
            if not any_pos.all():

                fill_idx = torch.argmax(sim_pos, dim=1)
                rows = (~any_pos).nonzero(as_tuple=False).squeeze(1)
                pos_mask_block[rows, fill_idx[rows]] = True

    pos_mask_block = pos_mask_block.to(device=device, dtype=torch.bool)

    pos_mask_full = torch.zeros((B, P + D), device=device, dtype=torch.bool)
    pos_mask_full[:, :P] = pos_mask_block

    # Contrast every anchor against all positives and decoys in the batch.
    pos_logits = logits.masked_fill(~pos_mask_full, float("-inf"))
    log_num = torch.logsumexp(pos_logits, dim=1)
    log_den = torch.logsumexp(logits, dim=1)
    loss = -(log_num - log_den).mean()

    # Penalize decoys whose cosine similarity exceeds the margin.
    margin = 0.2
    neg_sim = sim_dec
    neg_loss = F.relu(neg_sim - margin).mean()

    beta = 1.0
    loss = loss + beta * neg_loss

    with torch.no_grad():
        pos_mask_f = pos_mask_block.float()
        num_pos = pos_mask_f.sum(dim=1).clamp_min(1.0)
        pos_sim = (sim_pos * pos_mask_f).sum(1) / num_pos
        dec_sim = sim_dec.mean(dim=1)

        stats = {
            "loss": float(loss.item()),
            "pos_sim_mean": float(pos_sim.mean().item()),
            "neg_sim_mean": float(dec_sim.mean().item()),
            "pos_neg_gap": float((pos_sim.mean() - dec_sim.mean()).item()),
            "avg_num_positives": float(num_pos.mean().item()),
            "temperature": float(temperature),
            "decoy_weight": float(decoy_weight),
        }

    return loss, stats
