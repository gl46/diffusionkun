#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusionkun.dataset import Collator, TranslationJsonlDataset
from diffusionkun.metrics import corpus_metrics
from diffusionkun.model import DiffusionKun
from diffusionkun.sampling import diffuse_decode, diffuse_decode_length_beam, predict_length


def load_model(ckpt_path, spm_model, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    sp = spm.SentencePieceProcessor(model_file=spm_model or cfg['paths']['spm_model'])
    pad_id = ckpt.get('pad_id', sp.pad_id())
    mask_id = ckpt.get('mask_id', sp.piece_to_id('<mask>'))
    cfg['model']['vocab_size'] = sp.get_piece_size()
    model = DiffusionKun(
        vocab_size=cfg['model']['vocab_size'],
        d_model=cfg['model']['d_model'],
        encoder_layers=cfg['model']['encoder_layers'],
        decoder_layers=cfg['model']['decoder_layers'],
        heads=cfg['model']['heads'],
        ffn_dim=cfg['model']['ffn_dim'],
        dropout=0.0,
        max_src_len=cfg['model']['max_src_len'],
        max_tgt_len=cfg['model']['max_tgt_len'],
        pad_id=pad_id,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, sp, cfg, pad_id, mask_id


def decode_batch(sp, ids, pad_id):
    outs = []
    for row in ids.tolist():
        toks = [int(x) for x in row if int(x) != pad_id]
        outs.append(sp.decode(toks))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--spm_model', default='')
    ap.add_argument('--eval_jsonl', default='')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--oracle_length', action='store_true')
    ap.add_argument('--length_topk', type=int, default=1, help='try top-k predicted lengths when not using oracle length')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output', default='')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, sp, cfg, pad_id, mask_id = load_model(args.checkpoint, args.spm_model, device)
    eval_path = args.eval_jsonl or cfg['paths']['dev_jsonl']
    ds = TranslationJsonlDataset(
        path=eval_path,
        spm_model=args.spm_model or cfg['paths']['spm_model'],
        src_lang=cfg['data']['src_lang'],
        tgt_lang=cfg['data']['tgt_lang'],
        max_src_len=cfg['data']['max_src_len'],
        max_tgt_len=cfg['data']['max_tgt_len'],
    )
    if args.limit:
        ds.rows = ds.rows[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(pad_id), num_workers=2)

    srcs, refs, hyps = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, dynamic_ncols=True):
            src_ids = batch.src_ids.to(device)
            src_mask = batch.src_mask.to(device)
            if args.oracle_length:
                lengths = batch.tgt_lengths.to(device).clamp_max(cfg['model']['max_tgt_len'])
                y, _ = diffuse_decode(model, src_ids, src_mask, lengths, args.steps, mask_id, pad_id)
            elif args.length_topk > 1:
                y, _, _ = diffuse_decode_length_beam(
                    model, src_ids, src_mask, args.steps, mask_id, pad_id,
                    length_topk=args.length_topk, max_len=cfg['model']['max_tgt_len']
                )
            else:
                lengths = predict_length(model, src_ids, src_mask, min_len=1).clamp_max(cfg['model']['max_tgt_len'])
                y, _ = diffuse_decode(model, src_ids, src_mask, lengths, args.steps, mask_id, pad_id)
            hyps.extend(decode_batch(sp, y.cpu(), pad_id))
            srcs.extend(batch.raw_src)
            refs.extend(batch.raw_tgt)

    metrics = corpus_metrics(srcs, refs, hyps)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for s, r, h in zip(srcs, refs, hyps):
                f.write(json.dumps({'src': s, 'ref': r, 'hyp': h}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
