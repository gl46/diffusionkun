#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='clean jsonl with src/tgt')
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.input, 'r', encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            if args.limit and n >= args.limit:
                break
            obj = json.loads(line)
            fout.write(obj['src'].strip() + '\n')
            fout.write(obj['tgt'].strip() + '\n')
            n += 1
    print(f'wrote {2*n} lines to {args.output}')


if __name__ == '__main__':
    main()
