#!/usr/bin/env python
"""Audit a small boundary sample with a local OpenAI-compatible Qwen server.

This is for manual quality auditing around embedding thresholds, not for
full-corpus filtering.
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

SYSTEM_PROMPT = (
    'You are a strict bilingual parallel-corpus auditor. Judge whether the '
    'English text is a faithful, direct translation of the Chinese text. '
    'Return JSON only.'
)

USER_PROMPT = """Audit this zh-en parallel sentence pair for machine translation training.

Chinese source:
{src}

English target:
{tgt}

Return one compact JSON object only. Do not include explanations outside JSON.
Use exactly these keys:
{{
  "adequacy": 1,
  "fluency": 1,
  "omission": false,
  "hallucination": false,
  "numbers_preserved": true,
  "format_preserved": true,
  "decision": "keep"
}}

Scoring:
- adequacy and fluency are integers from 1 to 5.
- decision must be one of "keep", "reject", or "rewrite".
"""


def iter_jsonl(paths: List[str]) -> Iterable[dict]:
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_thresholds(path: str) -> Dict[str, float]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    return {k: float(v) for k, v in obj.get('thresholds', {}).items()}


def choose_boundary_sample(rows: Iterable[dict], n: int, thresholds: Dict[str, float], min_score: Optional[float], max_score: Optional[float]) -> List[dict]:
    candidates = []
    reject_t = thresholds.get('reject_threshold')
    silver_t = thresholds.get('silver_threshold')
    boundary_targets = [x for x in [reject_t, silver_t] if x is not None]

    for obj in rows:
        score = obj.get('emb_score')
        if score is None:
            continue
        score = float(score)
        if min_score is not None and score < min_score:
            continue
        if max_score is not None and score > max_score:
            continue
        if boundary_targets:
            distance = min(abs(score - target) for target in boundary_targets)
        else:
            distance = abs(score - 0.67)
        candidates.append((distance, score, obj))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [obj for _, _, obj in candidates[:n]]


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return {'parse_error': True, 'raw_response': text[:500]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {'parse_error': True, 'raw_response': text[:500]}


def normalize_audit(audit: dict) -> dict:
    out = {
        'adequacy': audit.get('adequacy'),
        'fluency': audit.get('fluency'),
        'omission': audit.get('omission', audit.get('has_omission')),
        'hallucination': audit.get('hallucination', audit.get('has_hallucination')),
        'numbers_preserved': audit.get('numbers_preserved'),
        'format_preserved': audit.get('format_preserved'),
        'decision': audit.get('decision'),
    }
    for key in ['adequacy', 'fluency']:
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            out[key] = None
    for key in ['omission', 'hallucination', 'numbers_preserved', 'format_preserved']:
        if isinstance(out[key], str):
            out[key] = out[key].strip().lower() == 'true'
    if out['decision'] not in {'keep', 'reject', 'rewrite'}:
        out['decision'] = 'reject'
    if audit.get('parse_error') or audit.get('error'):
        out['parse_error'] = audit.get('parse_error', False)
        out['error'] = audit.get('error')
        out['raw_response'] = audit.get('raw_response')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', action='append', required=True, help='Input JSONL. May be repeated for kept and rejected files.')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--base_url', default='http://127.0.0.1:8001/v1')
    ap.add_argument('--api_key', default='EMPTY')
    ap.add_argument('--model', default='qwen-audit')
    ap.add_argument('--thresholds_json', default='runs/goal_001/qwen3_embedding_calibration.json')
    ap.add_argument('--min_score', type=float, default=None)
    ap.add_argument('--max_score', type=float, default=None)
    ap.add_argument('--sleep', type=float, default=0.0)
    args = ap.parse_args()

    from openai import OpenAI

    thresholds = load_thresholds(args.thresholds_json) if Path(args.thresholds_json).exists() else {}
    sample = choose_boundary_sample(iter_jsonl(args.input), args.n, thresholds, args.min_score, args.max_score)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    with open(args.output, 'w', encoding='utf-8') as fout:
        for obj in tqdm(sample, dynamic_ncols=True):
            prompt = USER_PROMPT.format(src=obj.get('src', ''), tgt=obj.get('tgt', ''))
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt},
                    ],
                    temperature=0.0,
                    max_tokens=192,
                )
                raw = response.choices[0].message.content or ''
                audit = normalize_audit(extract_json(raw))
            except Exception as exc:
                audit = normalize_audit({'error': repr(exc)})
            out = dict(obj)
            out.update(audit)
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
            fout.flush()
            if args.sleep:
                time.sleep(args.sleep)

    print(json.dumps({'audited': len(sample), 'output': args.output}, ensure_ascii=False))


if __name__ == '__main__':
    main()
