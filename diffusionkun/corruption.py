from typing import Tuple

import torch


def corrupt_target(
    tgt_ids: torch.Tensor,
    pad_id: int,
    mask_id: int,
    vocab_size: int,
    noise_min: float = 0.05,
    noise_max: float = 0.995,
    mask_replace_prob: float = 0.9,
    random_replace_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized target corruption for conditional masked diffusion training.

    Args:
        tgt_ids: LongTensor [B, T]

    Returns:
        noisy_tgt: LongTensor [B, T]
        masked_positions: BoolTensor [B, T]
        valid_positions: BoolTensor [B, T]
        noise_ratio: FloatTensor [B]

    Notes:
        Source tokens are never corrupted; this function is target-only.
        Replacement policy is:
          - mask_replace_prob -> <mask>
          - random_replace_prob -> random non-special token
          - remaining probability -> keep original token but still compute masked loss
    """
    if not (0.0 <= mask_replace_prob <= 1.0 and 0.0 <= random_replace_prob <= 1.0):
        raise ValueError('replacement probabilities must be in [0, 1]')
    if mask_replace_prob + random_replace_prob > 1.0 + 1e-6:
        raise ValueError('mask_replace_prob + random_replace_prob must be <= 1')

    device = tgt_ids.device
    B, T = tgt_ids.shape
    valid = tgt_ids.ne(pad_id)
    valid_counts = valid.sum(dim=1)

    noisy = tgt_ids.clone()
    masked_positions = torch.zeros_like(tgt_ids, dtype=torch.bool)

    noise_ratio = noise_min + (noise_max - noise_min) * torch.rand(B, device=device)
    mask_counts = torch.round(noise_ratio * valid_counts.float()).long()
    mask_counts = torch.where(valid_counts > 0, mask_counts.clamp_min(1), torch.zeros_like(mask_counts))
    max_k = int(mask_counts.max().item()) if B > 0 else 0
    if max_k == 0:
        return noisy, masked_positions, valid, noise_ratio

    # Pick the lowest random scores among valid positions for each row.
    scores = torch.rand(B, T, device=device)
    scores = scores.masked_fill(~valid, 2.0)
    chosen = torch.topk(scores, k=max_k, dim=1, largest=False).indices  # [B, max_k]
    ranks = torch.arange(max_k, device=device).unsqueeze(0)
    chosen_valid = ranks < mask_counts.unsqueeze(1)
    masked_positions.scatter_(1, chosen, chosen_valid)

    r = torch.rand(B, T, device=device)
    use_mask = masked_positions & (r < mask_replace_prob)
    use_random = masked_positions & (r >= mask_replace_prob) & (r < mask_replace_prob + random_replace_prob)

    noisy[use_mask] = mask_id

    if use_random.any():
        # Avoid special ids 0..3 by default. Adjust if your tokenizer differs.
        low_random_id = min(4, vocab_size - 1)
        random_tokens = torch.randint(
            low=low_random_id,
            high=vocab_size,
            size=tgt_ids.shape,
            device=device,
        )
        noisy = torch.where(use_random, random_tokens, noisy)

    return noisy, masked_positions, valid, noise_ratio
