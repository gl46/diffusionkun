#!/usr/bin/env python
import argparse
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--dev_size', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    lines = [line for line in open(args.input, 'r', encoding='utf-8') if line.strip()]
    random.shuffle(lines)
    dev = lines[:args.dev_size]
    train = lines[args.dev_size:]

    Path(args.train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train, 'w', encoding='utf-8') as f:
        f.writelines(train)
    with open(args.dev, 'w', encoding='utf-8') as f:
        f.writelines(dev)
    print({'train': len(train), 'dev': len(dev)})


if __name__ == '__main__':
    main()
