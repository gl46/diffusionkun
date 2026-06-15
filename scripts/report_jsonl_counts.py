#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Count JSONL rows and optionally count a field.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--field', default='reject_reason')
    ap.add_argument('--top', type=int, default=30)
    args = ap.parse_args()

    path = Path(args.input)
    counts = Counter()
    rows = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                obj = json.loads(line)
                counts[str(obj.get(args.field, '<missing>'))] += 1
            except json.JSONDecodeError:
                counts['<json_error>'] += 1

    print(f'path\t{path}')
    print(f'rows\t{rows}')
    print(f'field\t{args.field}')
    for key, value in counts.most_common(args.top):
        print(f'{key}\t{value}\t{value / max(rows, 1):.6f}')


if __name__ == '__main__':
    main()
