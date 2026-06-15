#!/usr/bin/env python
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    lines = [line for line in open(args.input, 'r', encoding='utf-8') if line.strip()]
    random.shuffle(lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.writelines(lines[:args.n])
    print(f'wrote {min(args.n, len(lines))} lines to {args.output}')


if __name__ == '__main__':
    main()
