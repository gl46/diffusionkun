#!/usr/bin/env python
import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from diffusionkun.model import DiffusionKun
from diffusionkun.sampling import diffuse_decode, diffuse_decode_length_beam, predict_length


def encode_src(sp, text, src_lang='zh', tgt_lang='en', max_src_len=128):
    ids = [sp.piece_to_id(f'<src_{src_lang}>'), sp.piece_to_id(f'<tgt_{tgt_lang}>')] + sp.encode(text, out_type=int)
    return ids[:max_src_len]


def decode_ids(sp, ids, pad_id):
    toks = [int(x) for x in ids if int(x) != pad_id]
    return sp.decode(toks)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--spm_model', default='')
    ap.add_argument('--text', default='')
    ap.add_argument('--input_file', default='')
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--length', type=int, default=0, help='force target length; 0 = predicted')
    ap.add_argument('--length_topk', type=int, default=1, help='try top-k predicted lengths when --length=0')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, sp, cfg, pad_id, mask_id = load_model(args.checkpoint, args.spm_model, device)

    if args.input_file:
        texts = [x.strip() for x in open(args.input_file, 'r', encoding='utf-8') if x.strip()]
    else:
        texts = [args.text]

    for text in texts:
        ids = encode_src(sp, text, cfg['data']['src_lang'], cfg['data']['tgt_lang'], cfg['model']['max_src_len'])
        src_ids = torch.tensor([ids], dtype=torch.long, device=device)
        src_mask = src_ids.ne(pad_id)
        if args.length > 0:
            lengths = torch.tensor([args.length], dtype=torch.long, device=device)
            y, conf = diffuse_decode(model, src_ids, src_mask, lengths, args.steps, mask_id, pad_id)
        elif args.length_topk > 1:
            y, lengths, conf = diffuse_decode_length_beam(
                model, src_ids, src_mask, args.steps, mask_id, pad_id,
                length_topk=args.length_topk, max_len=cfg['model']['max_tgt_len']
            )
        else:
            lengths = predict_length(model, src_ids, src_mask, min_len=1).clamp_max(cfg['model']['max_tgt_len'])
            y, conf = diffuse_decode(model, src_ids, src_mask, lengths, args.steps, mask_id, pad_id)
        print(decode_ids(sp, y[0].detach().cpu().tolist(), pad_id))


if __name__ == '__main__':
    main()
