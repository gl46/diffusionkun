#!/usr/bin/env python
"""Semantic filtering for zh-en parallel data with sentence embeddings.

Default model is Qwen/Qwen3-Embedding-0.6B. Qwen thresholds are calibrated from
the current rule-cleaned corpus instead of reusing LaBSE constants.
"""
import argparse
import inspect
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_EMB_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
ALIGNMENT_INSTRUCTION = 'Represent this sentence for cross-lingual translation alignment.'
QUANTILES = [
    ('p01', 0.01),
    ('p05', 0.05),
    ('p10', 0.10),
    ('p25', 0.25),
    ('p40', 0.40),
    ('p50', 0.50),
    ('p75', 0.75),
    ('p90', 0.90),
    ('p95', 0.95),
    ('p99', 0.99),
]
LABSE_PRESET = {
    'reject_threshold': 0.62,
    'bronze_threshold': 0.62,
    'silver_threshold': 0.72,
    'gold_threshold': 0.82,
}


def iter_jsonl(path: str, limit: int = 0) -> Iterable[dict]:
    seen = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if limit and seen >= limit:
                break
            seen += 1
            yield json.loads(line)


def batched(rows: Iterable[dict], batch_size: int) -> Iterable[List[dict]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def reservoir_sample(path: str, size: int, seed: int) -> Tuple[List[dict], int]:
    rng = random.Random(seed)
    sample = []
    seen = 0
    for obj in iter_jsonl(path):
        seen += 1
        if len(sample) < size:
            sample.append(obj)
            continue
        j = rng.randint(0, seen - 1)
        if j < size:
            sample[j] = obj
    return sample, seen


def get_torch_dtype(precision: str, device: str):
    import torch

    if precision == 'auto':
        if device.startswith('cuda'):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if precision == 'bf16':
        return torch.bfloat16
    if precision == 'fp16':
        return torch.float16
    return torch.float32


def load_embedding_model(model_name: str, device: Optional[str], precision: str):
    import torch
    from sentence_transformers import SentenceTransformer

    resolved_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = get_torch_dtype(precision, resolved_device)
    model_kwargs = {}
    if resolved_device.startswith('cuda') and dtype in (torch.float16, torch.bfloat16):
        model_kwargs['torch_dtype'] = dtype

    try:
        model = SentenceTransformer(model_name, device=resolved_device, model_kwargs=model_kwargs)
    except TypeError:
        model = SentenceTransformer(model_name, device=resolved_device)
        if resolved_device.startswith('cuda') and dtype == torch.float16:
            model.half()
        elif resolved_device.startswith('cuda') and dtype == torch.bfloat16:
            model.to(dtype=torch.bfloat16)
    return model, resolved_device, str(dtype).replace('torch.', '')


def encode_texts(model, texts: List[str], batch_size: int, instruction: Optional[str]) -> np.ndarray:
    kwargs = {
        'batch_size': batch_size,
        'normalize_embeddings': True,
        'show_progress_bar': False,
    }
    if instruction:
        try:
            sig = inspect.signature(model.encode)
            if 'prompt' in sig.parameters:
                return np.asarray(model.encode(texts, prompt=instruction, **kwargs), dtype=np.float32)
        except (TypeError, ValueError):
            pass
        texts = [f'{instruction}\n{text}' for text in texts]
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def score_batches(model, rows: List[dict], batch_size: int, instruction: Optional[str]) -> np.ndarray:
    scores = []
    for batch in tqdm(list(batched(rows, batch_size)), desc='embedding calibration', dynamic_ncols=True):
        srcs = [x.get('src', '') for x in batch]
        tgts = [x.get('tgt', '') for x in batch]
        src_emb = encode_texts(model, srcs, batch_size, instruction)
        tgt_emb = encode_texts(model, tgts, batch_size, instruction)
        scores.append(np.sum(src_emb * tgt_emb, axis=1))
    if not scores:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(scores).astype(np.float32)


def score_negative_pairs(model, rows: List[dict], batch_size: int, instruction: Optional[str], seed: int) -> np.ndarray:
    if len(rows) < 2:
        return np.asarray([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(rows))
    if np.any(permutation == np.arange(len(rows))):
        permutation = np.roll(np.arange(len(rows)), 1)

    scores = []
    for start in tqdm(range(0, len(rows), batch_size), desc='negative calibration', dynamic_ncols=True):
        end = min(start + batch_size, len(rows))
        srcs = [rows[i].get('src', '') for i in range(start, end)]
        tgts = [rows[int(permutation[i])].get('tgt', '') for i in range(start, end)]
        src_emb = encode_texts(model, srcs, batch_size, instruction)
        tgt_emb = encode_texts(model, tgts, batch_size, instruction)
        scores.append(np.sum(src_emb * tgt_emb, axis=1))
    return np.concatenate(scores).astype(np.float32)


def quantile_report(values: np.ndarray) -> Dict[str, float]:
    if len(values) == 0:
        raise ValueError('Cannot compute quantiles for an empty score array.')
    return {name: round(float(np.quantile(values, q)), 6) for name, q in QUANTILES}


def choose_auto_thresholds(pos_q: Dict[str, float], neg_q: Dict[str, float], strategy: str) -> Tuple[Dict[str, float], str, Dict[str, float]]:
    strict_reject = max(neg_q['p99'] + 0.02, pos_q['p05'])
    balanced_reject = pos_q['p05']
    candidates = {
        'strict_reject_threshold': round(float(strict_reject), 6),
        'balanced_reject_threshold': round(float(balanced_reject), 6),
        'negative_p99_plus_margin': round(float(neg_q['p99'] + 0.02), 6),
        'positive_p05': round(float(pos_q['p05']), 6),
    }

    if strategy == 'strict':
        reject = strict_reject
        mode = 'strict_negative_p99_plus_margin_or_positive_p05'
    elif strategy == 'balanced':
        # OPUS contains many short/template-like rows, so shuffled negatives can
        # have a high p99. The balanced strategy keeps the requested strict
        # candidate in the report but caps the actual reject line at positive p05
        # to avoid rejecting a large fraction of true translations.
        reject = balanced_reject
        mode = 'balanced_positive_p05_cap'
    else:
        raise ValueError(f'Unknown threshold strategy: {strategy}')

    thresholds = {
        'reject_threshold': round(float(reject), 6),
        'bronze_threshold': round(float(reject), 6),
        'silver_threshold': round(float(pos_q['p40']), 6),
        'gold_threshold': round(float(pos_q['p75']), 6),
    }

    if not (
        thresholds['reject_threshold']
        <= thresholds['silver_threshold']
        <= thresholds['gold_threshold']
    ):
        thresholds = {
            'reject_threshold': round(float(pos_q['p10']), 6),
            'bronze_threshold': round(float(pos_q['p10']), 6),
            'silver_threshold': round(float(pos_q['p50']), 6),
            'gold_threshold': round(float(pos_q['p75']), 6),
        }
        mode = f'{strategy}_fallback_positive_quantiles'

    return thresholds, mode, candidates


def manual_thresholds(args) -> Dict[str, float]:
    if args.keep_threshold is not None and args.reject_threshold is None:
        args.reject_threshold = args.keep_threshold
    if args.bronze is not None and args.reject_threshold is None:
        args.reject_threshold = args.bronze

    if all(x is None for x in [args.reject_threshold, args.bronze, args.silver, args.gold]):
        if args.model == 'sentence-transformers/LaBSE':
            return dict(LABSE_PRESET)
        raise ValueError(
            'Non-LaBSE models require --auto_thresholds or explicit '
            '--reject_threshold/--silver/--gold values.'
        )

    reject = args.reject_threshold
    bronze = args.bronze
    silver = args.silver
    gold = args.gold
    if reject is None:
        reject = bronze
    if bronze is None:
        bronze = reject
    if silver is None or gold is None or reject is None:
        raise ValueError('Manual thresholds need --reject_threshold (or --bronze), --silver, and --gold.')

    thresholds = {
        'reject_threshold': float(reject),
        'bronze_threshold': float(bronze),
        'silver_threshold': float(silver),
        'gold_threshold': float(gold),
    }
    if not (
        thresholds['reject_threshold']
        <= thresholds['bronze_threshold']
        <= thresholds['silver_threshold']
        <= thresholds['gold_threshold']
    ):
        raise ValueError(f'Thresholds must be monotonic: {thresholds}')
    return thresholds


def bucket_score(score: float, thresholds: Dict[str, float]) -> str:
    if score < thresholds['reject_threshold']:
        return 'reject'
    if score >= thresholds['gold_threshold']:
        return 'gold'
    if score >= thresholds['silver_threshold']:
        return 'silver'
    return 'bronze'


def write_calibration_reports(args, report: Dict):
    json_path = Path(args.calibration_json)
    txt_path = Path(args.calibration_txt)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        f"model\t{report['model']}",
        f"sampled_pairs\t{report['sampled_pairs']}",
        f"total_input_rows_seen\t{report['total_input_rows_seen']}",
        f"threshold_mode\t{report['threshold_mode']}",
        '',
        '[positive_quantiles]',
    ]
    lines += [f'{k}\t{v}' for k, v in report['positive_quantiles'].items()]
    lines += ['', '[negative_quantiles]']
    lines += [f'{k}\t{v}' for k, v in report['negative_quantiles'].items()]
    if 'threshold_candidates' in report:
        lines += ['', '[threshold_candidates]']
        lines += [f'{k}\t{v}' for k, v in report['threshold_candidates'].items()]
    lines += ['', '[thresholds]']
    lines += [f'{k}\t{v}' for k, v in report['thresholds'].items()]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def calibrate(args, model, instruction: Optional[str]) -> Dict:
    sample, total_seen = reservoir_sample(args.input, args.calibration_size, args.seed)
    if len(sample) < 2:
        raise ValueError('Need at least two rows for embedding threshold calibration.')
    pos_scores = score_batches(model, sample, args.batch_size, instruction)
    neg_scores = score_negative_pairs(model, sample, args.batch_size, instruction, args.seed + 17)
    pos_q = quantile_report(pos_scores)
    neg_q = quantile_report(neg_scores)
    thresholds, mode, candidates = choose_auto_thresholds(pos_q, neg_q, args.threshold_strategy)
    report = {
        'model': args.model,
        'instruction': instruction or '',
        'sampled_pairs': len(sample),
        'total_input_rows_seen': total_seen,
        'positive_quantiles': pos_q,
        'negative_quantiles': neg_q,
        'threshold_candidates': candidates,
        'thresholds': thresholds,
        'threshold_mode': mode,
    }
    write_calibration_reports(args, report)
    return report


def write_filter_report(path: str, rows_seen: int, kept: int, rejected: int, buckets: Counter, thresholds: Dict[str, float], args):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'model\t{args.model}',
        f'rows_seen\t{rows_seen}',
        f'kept\t{kept}',
        f'rejected\t{rejected}',
        f'instruction\t{"" if args.no_instruction else ALIGNMENT_INSTRUCTION}',
        '',
        '[thresholds]',
    ]
    lines += [f'{k}\t{v}' for k, v in thresholds.items()]
    lines += ['', '[buckets]']
    for key in ['gold', 'silver', 'bronze', 'reject']:
        lines.append(f'{key}\t{buckets.get(key, 0)}')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_filter(args, model, instruction: Optional[str], thresholds: Dict[str, float]):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reject).parent.mkdir(parents=True, exist_ok=True)

    rows_seen = kept = rejected = 0
    buckets = Counter()
    with open(args.output, 'w', encoding='utf-8') as fout, open(args.reject, 'w', encoding='utf-8') as frej:
        batch_iter = batched(iter_jsonl(args.input, args.limit), args.batch_size)
        for batch in tqdm(batch_iter, desc='embedding filter', dynamic_ncols=True):
            srcs = [x.get('src', '') for x in batch]
            tgts = [x.get('tgt', '') for x in batch]
            src_emb = encode_texts(model, srcs, args.batch_size, instruction)
            tgt_emb = encode_texts(model, tgts, args.batch_size, instruction)
            scores = np.sum(src_emb * tgt_emb, axis=1)
            for obj, score in zip(batch, scores):
                rows_seen += 1
                score = round(float(score), 6)
                emb_bucket = bucket_score(score, thresholds)
                buckets[emb_bucket] += 1
                out = dict(obj)
                out['emb_score'] = score
                out['emb_bucket'] = emb_bucket
                out['emb_model'] = args.model
                if emb_bucket == 'reject':
                    out['reject_reason'] = 'low_embedding_score'
                    frej.write(json.dumps(out, ensure_ascii=False) + '\n')
                    rejected += 1
                else:
                    fout.write(json.dumps(out, ensure_ascii=False) + '\n')
                    kept += 1

    write_filter_report(args.report, rows_seen, kept, rejected, buckets, thresholds, args)
    print(json.dumps({
        'rows_seen': rows_seen,
        'kept': kept,
        'rejected': rejected,
        'buckets': dict(buckets),
        'thresholds': thresholds,
        'model': args.model,
    }, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--reject', required=True)
    ap.add_argument('--report', default='runs/goal_001/embed_quality_report.txt')
    ap.add_argument('--model', default=DEFAULT_EMB_MODEL)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--device', default=None)
    ap.add_argument('--precision', choices=['auto', 'fp32', 'fp16', 'bf16'], default='auto')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_instruction', action='store_true')

    ap.add_argument('--auto_thresholds', action='store_true')
    ap.add_argument(
        '--threshold_strategy',
        choices=['strict', 'balanced'],
        default='strict',
        help='strict follows max(negative_p99+0.02, positive_p05); balanced caps reject at positive_p05 while reporting both candidates.',
    )
    ap.add_argument('--calibration_size', type=int, default=50000)
    ap.add_argument('--calibration_json', default='runs/goal_001/qwen3_embedding_calibration.json')
    ap.add_argument('--calibration_txt', default='runs/goal_001/qwen3_embedding_calibration.txt')

    ap.add_argument('--reject_threshold', type=float, default=None)
    ap.add_argument('--keep_threshold', type=float, default=None, help='Backward-compatible alias for --reject_threshold.')
    ap.add_argument('--bronze', type=float, default=None)
    ap.add_argument('--silver', type=float, default=None)
    ap.add_argument('--gold', type=float, default=None)
    args = ap.parse_args()

    instruction = None if args.no_instruction else ALIGNMENT_INSTRUCTION
    model, device, dtype = load_embedding_model(args.model, args.device, args.precision)
    print(json.dumps({'model': args.model, 'device': device, 'dtype': dtype, 'instruction': instruction or ''}, ensure_ascii=False))

    if args.auto_thresholds:
        calibration = calibrate(args, model, instruction)
        thresholds = calibration['thresholds']
    else:
        thresholds = manual_thresholds(args)

    run_filter(args, model, instruction, thresholds)


if __name__ == '__main__':
    main()
