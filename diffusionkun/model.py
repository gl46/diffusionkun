import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class NoiseEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, noise_ratio: torch.Tensor) -> torch.Tensor:
        return self.net(noise_ratio[:, None])


class DiffusionKun(nn.Module):
    """Encoder-decoder conditional masked diffusion translator.

    Decoder self-attention is bidirectional because no causal target mask is used.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        encoder_layers: int,
        decoder_layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        max_src_len: int,
        max_tgt_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.src_pos_emb = nn.Embedding(max_src_len, d_model)
        self.tgt_pos_emb = nn.Embedding(max_tgt_len, d_model)
        self.noise_emb = NoiseEmbedding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)

        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.length_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max_tgt_len + 1),
        )

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Conservative init for scratch training.
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'token_emb' not in name and 'lm_head' not in name:
                nn.init.xavier_uniform_(p)

    def _positions(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.arange(length, device=device).unsqueeze(0)

    def encode(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S = src_ids.shape
        if S > self.max_src_len:
            raise ValueError(f'src length {S} > max_src_len {self.max_src_len}')
        pos = self._positions(S, src_ids.device)
        x = self.token_emb(src_ids) + self.src_pos_emb(pos)
        x = self.dropout(x)
        memory = self.encoder(x, src_key_padding_mask=~src_mask)

        # Mean pool non-padding encoder states for length prediction.
        masked = memory * src_mask.unsqueeze(-1).to(memory.dtype)
        pooled = masked.sum(dim=1) / src_mask.sum(dim=1, keepdim=True).clamp_min(1).to(memory.dtype)
        length_logits = self.length_head(pooled)
        return memory, length_logits

    def forward(
        self,
        src_ids: torch.Tensor,
        src_mask: torch.Tensor,
        noisy_tgt_ids: torch.Tensor,
        tgt_valid_mask: torch.Tensor,
        noise_ratio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = noisy_tgt_ids.shape
        if T > self.max_tgt_len:
            raise ValueError(f'target length {T} > max_tgt_len {self.max_tgt_len}')

        memory, length_logits = self.encode(src_ids, src_mask)

        pos = self._positions(T, noisy_tgt_ids.device)
        nemb = self.noise_emb(noise_ratio).unsqueeze(1)
        y = self.token_emb(noisy_tgt_ids) + self.tgt_pos_emb(pos) + nemb
        y = self.dropout(y)

        # Important: no tgt_mask, so target self-attention is bidirectional.
        dec = self.decoder(
            tgt=y,
            memory=memory,
            tgt_key_padding_mask=~tgt_valid_mask,
            memory_key_padding_mask=~src_mask,
        )
        dec = self.final_norm(dec)
        logits = self.lm_head(dec)
        return logits, length_logits
