from __future__ import annotations
import math
from typing import Tuple

import torch


@torch.no_grad()
def diffuse_decode(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    lengths: torch.Tensor,
    steps: int,
    mask_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Iterative confidence-based masked denoising.

    Args:
        lengths: LongTensor [B], target lengths to generate.

    Returns:
        y: LongTensor [B, max_len]
        avg_conf: FloatTensor [B]
    """
    device = src_ids.device
    B = src_ids.size(0)
    max_len = int(lengths.max().item())
    max_len = max(1, max_len)

    y = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    valid = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
    y[valid] = mask_id

    last_conf = torch.zeros((B, max_len), device=device)

    for step in range(steps):
        progress = (step + 1) / steps
        # Cosine schedule: many masks early, none at the final step.
        next_mask_ratio = math.cos(progress * math.pi / 2)

        current_noise_ratio = y.eq(mask_id).float().sum(dim=1) / lengths.clamp_min(1).float()
        logits, _ = model(
            src_ids=src_ids,
            src_mask=src_mask,
            noisy_tgt_ids=y,
            tgt_valid_mask=valid,
            noise_ratio=current_noise_ratio,
        )
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        last_conf = conf

        filled = torch.where(valid, pred, torch.full_like(pred, pad_id))
        new_y = filled.clone()

        # T is small (<=128), so this row loop is fine for sampling/debug.
        # Training-time corruption is vectorized separately.
        for b in range(B):
            n_valid = int(lengths[b].item())
            if n_valid <= 0:
                continue
            n_mask_next = int(round(next_mask_ratio * n_valid))
            if n_mask_next <= 0:
                continue
            low_conf_pos = torch.argsort(conf[b, :n_valid])[:n_mask_next]
            new_y[b, low_conf_pos] = mask_id

        y = new_y
        y[~valid] = pad_id

    # Fill any remaining masks with one final argmax pass.
    if y.eq(mask_id).any():
        current_noise_ratio = y.eq(mask_id).float().sum(dim=1) / lengths.clamp_min(1).float()
        logits, _ = model(
            src_ids=src_ids,
            src_mask=src_mask,
            noisy_tgt_ids=y,
            tgt_valid_mask=valid,
            noise_ratio=current_noise_ratio,
        )
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        y = torch.where(valid, pred, torch.full_like(pred, pad_id))
        last_conf = conf

    avg_conf = (last_conf * valid.float()).sum(dim=1) / lengths.clamp_min(1).float()
    return y, avg_conf


@torch.no_grad()
def predict_length_topk(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    k: int = 1,
    min_len: int = 1,
    max_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-k predicted target lengths and log-probs.

    Returns:
        lengths: LongTensor [B, k]
        scores: FloatTensor [B, k]
    """
    _, length_logits = model.encode(src_ids, src_mask)
    length_logits = length_logits.clone()
    length_logits[:, :min_len] = -1e9
    if max_len is not None and max_len + 1 < length_logits.size(1):
        length_logits[:, max_len + 1:] = -1e9
    log_probs = torch.log_softmax(length_logits, dim=-1)
    scores, lengths = torch.topk(log_probs, k=min(k, log_probs.size(1)), dim=-1)
    return lengths.clamp_min(min_len), scores


@torch.no_grad()
def predict_length(model, src_ids: torch.Tensor, src_mask: torch.Tensor, min_len: int = 1) -> torch.Tensor:
    lengths, _ = predict_length_topk(model, src_ids, src_mask, k=1, min_len=min_len)
    return lengths[:, 0]


@torch.no_grad()
def diffuse_decode_length_beam(
    model,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    steps: int,
    mask_id: int,
    pad_id: int,
    length_topk: int = 3,
    min_len: int = 1,
    max_len: int | None = None,
    length_score_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Try top-k predicted lengths and keep the highest confidence candidate.

    This is a cheap Mask-Predict style length beam. It usually fixes many
    predicted-length failures without changing the model.

    Returns:
        y: LongTensor [B, T_best_padded]
        selected_lengths: LongTensor [B]
        selected_scores: FloatTensor [B]
    """
    device = src_ids.device
    B = src_ids.size(0)
    max_len = max_len or getattr(model, 'max_tgt_len', None)
    cand_lengths, len_logp = predict_length_topk(
        model, src_ids, src_mask, k=length_topk, min_len=min_len, max_len=max_len
    )

    cand_ys = []
    cand_scores = []
    for j in range(cand_lengths.size(1)):
        lengths = cand_lengths[:, j]
        y, conf = diffuse_decode(model, src_ids, src_mask, lengths, steps, mask_id, pad_id)
        score = conf + length_score_weight * len_logp[:, j]
        cand_ys.append(y)
        cand_scores.append(score)

    scores = torch.stack(cand_scores, dim=1)  # [B, K]
    best = scores.argmax(dim=1)
    best_lengths = cand_lengths.gather(1, best[:, None]).squeeze(1)
    best_scores = scores.gather(1, best[:, None]).squeeze(1)

    out_max_len = int(best_lengths.max().item())
    out = torch.full((B, max(1, out_max_len)), pad_id, dtype=torch.long, device=device)
    for b in range(B):
        yb = cand_ys[int(best[b].item())][b, : int(best_lengths[b].item())]
        out[b, : yb.numel()] = yb
    return out, best_lengths, best_scores
