#!/usr/bin/env python
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def count_field(path: Path, field: str) -> Counter:
    counts = Counter()
    if not path.exists():
        return counts
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            counts[str(obj.get(field, '<missing>'))] += 1
    return counts


def parse_metrics(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    text = path.read_text(encoding='utf-8')
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def fmt_metrics(metrics: Optional[dict]) -> str:
    if not metrics:
        return '<missing>'
    keys = ['bleu', 'chrf', 'number_match_rate', 'url_match_rate', 'bad_prefix_rate']
    return ', '.join(f'{key}={float(metrics[key]):.4f}' for key in keys if key in metrics)


def print_samples(path: Path, title: str, n: int):
    print(f'\n[{title}]')
    if not path.exists():
        print('<missing>')
        return
    printed = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            hyp = obj.get('hyp', '')
            mask_flag = ' MASK_PRESENT' if '<mask>' in hyp else ''
            print(f'{printed + 1}. SRC: {obj.get("src", "")}')
            print(f'   REF: {obj.get("ref", "")}')
            print(f'   HYP: {hyp}{mask_flag}')
            printed += 1
            if printed >= n:
                break
    if printed == 0:
        print('<empty>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', default='runs/goal_001')
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--samples', type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    raw = data_dir / 'raw/opus100_zh_en.raw.jsonl'
    rule_clean = data_dir / 'cleaned/zh_en.rule.clean.jsonl'
    rule_rejected = data_dir / 'rejected/zh_en.rule.rejected.jsonl'
    embed_clean = data_dir / 'cleaned/zh_en.embed.clean.jsonl'
    embed_rejected = data_dir / 'rejected/zh_en.embed.rejected.jsonl'

    print('[Goal-001 Summary]')
    print(f'raw_rows\t{count_rows(raw)}')
    print(f'rule_clean_kept\t{count_rows(rule_clean)}')
    print(f'rule_clean_rejected\t{count_rows(rule_rejected)}')
    print(f'embed_clean_kept\t{count_rows(embed_clean)}')
    print(f'embed_rejected\t{count_rows(embed_rejected)}')

    print('\n[top_rule_reject_reasons]')
    for key, value in count_field(rule_rejected, 'reject_reason').most_common(10):
        print(f'{key}\t{value}')

    print('\n[embedding_buckets]')
    bucket_counts = count_field(embed_clean, 'emb_bucket')
    bucket_counts.update(count_field(embed_rejected, 'emb_bucket'))
    for key in ['gold', 'silver', 'bronze', 'reject']:
        print(f'{key}\t{bucket_counts.get(key, 0)}')

    cal_path = run_dir / 'qwen3_embedding_calibration.json'
    print('\n[embedding_thresholds]')
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding='utf-8'))
        for key, value in cal.get('thresholds', {}).items():
            print(f'{key}\t{value}')
    else:
        print('<missing>')

    print('\n[metrics]')
    print(f'30m_oracle\t{fmt_metrics(parse_metrics(run_dir / "eval_30m_oracle.txt"))}')
    print(f'30m_top1\t{fmt_metrics(parse_metrics(run_dir / "eval_30m_top1.txt"))}')
    print(f'30m_top3\t{fmt_metrics(parse_metrics(run_dir / "eval_30m_top3.txt"))}')
    print(f'100m_oracle\t{fmt_metrics(parse_metrics(run_dir / "eval_100m_oracle.txt"))}')
    print(f'100m_top3\t{fmt_metrics(parse_metrics(run_dir / "eval_100m_top3.txt"))}')

    print_samples(run_dir / '30m_top3.jsonl', '30M top-3 samples', args.samples)
    print_samples(run_dir / '100m_top3.jsonl', '100M top-3 samples', args.samples)


if __name__ == '__main__':
    main()
