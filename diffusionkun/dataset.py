import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    src_ids: torch.Tensor
    src_mask: torch.Tensor
    tgt_ids: torch.Tensor
    tgt_mask: torch.Tensor
    tgt_lengths: torch.Tensor
    raw_src: List[str]
    raw_tgt: List[str]


class TranslationJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        spm_model: str,
        src_lang: str = 'zh',
        tgt_lang: str = 'en',
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.path = path
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_tag = f'<src_{src_lang}>'
        self.tgt_tag = f'<tgt_{tgt_lang}>'

        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        self.pad_id = self.sp.pad_id()
        if self.pad_id < 0:
            raise ValueError('SentencePiece model must have pad_id >= 0.')
        self.mask_id = self.sp.piece_to_id('<mask>')
        if self.mask_id < 0:
            raise ValueError('SentencePiece model must include <mask>.')

        self.src_tag_id = self.sp.piece_to_id(self.src_tag)
        self.tgt_tag_id = self.sp.piece_to_id(self.tgt_tag)
        if self.src_tag_id < 0 or self.tgt_tag_id < 0:
            raise ValueError(f'SentencePiece model must include {self.src_tag} and {self.tgt_tag}.')

        self.rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                src = obj.get('src', '').strip()
                tgt = obj.get('tgt', '').strip()
                if src and tgt:
                    self.rows.append({'src': src, 'tgt': tgt})
        if not self.rows:
            raise ValueError(f'No valid rows found in {path}')

    def __len__(self):
        return len(self.rows)

    def encode_src(self, text: str) -> List[int]:
        ids = [self.src_tag_id, self.tgt_tag_id] + self.sp.encode(text, out_type=int)
        return ids[:self.max_src_len]

    def encode_tgt(self, text: str) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        return ids[:self.max_tgt_len]

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        src_ids = self.encode_src(row['src'])
        tgt_ids = self.encode_tgt(row['tgt'])
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'raw_src': row['src'],
            'raw_tgt': row['tgt'],
        }


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples: List[Dict]) -> Batch:
        max_src = max(len(x['src_ids']) for x in examples)
        max_tgt = max(len(x['tgt_ids']) for x in examples)
        B = len(examples)

        src_ids = torch.full((B, max_src), self.pad_id, dtype=torch.long)
        tgt_ids = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)
        tgt_lengths = torch.zeros(B, dtype=torch.long)

        raw_src, raw_tgt = [], []
        for i, ex in enumerate(examples):
            s = torch.tensor(ex['src_ids'], dtype=torch.long)
            t = torch.tensor(ex['tgt_ids'], dtype=torch.long)
            src_ids[i, :len(s)] = s
            tgt_ids[i, :len(t)] = t
            tgt_lengths[i] = len(t)
            raw_src.append(ex['raw_src'])
            raw_tgt.append(ex['raw_tgt'])

        src_mask = src_ids.ne(self.pad_id)
        tgt_mask = tgt_ids.ne(self.pad_id)
        return Batch(src_ids, src_mask, tgt_ids, tgt_mask, tgt_lengths, raw_src, raw_tgt)
