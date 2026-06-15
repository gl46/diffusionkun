#!/usr/bin/env python
"""Download/stream OPUS-100 into DiffusionKun raw jsonl.

Example:
  python scripts/download_opus100.py --pair en-zh --src_lang zh --tgt_lang en \
    --split train --output data/raw/opus100_zh_en.raw.jsonl --limit 1000000
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='en-zh', help='OPUS-100 config, e.g. en-zh')
    ap.add_argument('--src_lang', default='zh')
    ap.add_argument('--tgt_lang', default='en')
    ap.add_argument('--split', default='train')
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--streaming', action='store_true')
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('Helsinki-NLP/opus-100', args.pair, split=args.split, streaming=args.streaming)

    n = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for ex in tqdm(ds, dynamic_ncols=True):
            tr = ex.get('translation', ex)
            if args.src_lang not in tr or args.tgt_lang not in tr:
                raise KeyError(f'Expected {args.src_lang}/{args.tgt_lang} in translation keys={list(tr.keys())}')
            src = (tr[args.src_lang] or '').strip()
            tgt = (tr[args.tgt_lang] or '').strip()
            if not src or not tgt:
                continue
            f.write(json.dumps({'src': src, 'tgt': tgt}, ensure_ascii=False) + '\n')
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f'wrote {n} rows to {args.output}')


if __name__ == '__main__':
    main()
