import re
from typing import Dict, List

import sacrebleu

NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
URL_RE = re.compile(r"https?://[^\s]+")

BAD_PREFIXES = [
    'the translation is',
    'here is the translation',
    'it can be translated as',
    'translation:',
    '译文如下',
    '这句话可以翻译为',
]


def _norm_nums(text: str):
    return sorted(x.replace(',', '') for x in NUM_RE.findall(text))


def number_match_rate(srcs: List[str], hyps: List[str]) -> float:
    total = 0
    ok = 0
    for s, h in zip(srcs, hyps):
        nums = _norm_nums(s)
        if not nums:
            continue
        total += 1
        if nums == _norm_nums(h):
            ok += 1
    return ok / total if total else 1.0


def url_match_rate(srcs: List[str], hyps: List[str]) -> float:
    total = 0
    ok = 0
    for s, h in zip(srcs, hyps):
        urls = sorted(URL_RE.findall(s))
        if not urls:
            continue
        total += 1
        if urls == sorted(URL_RE.findall(h)):
            ok += 1
    return ok / total if total else 1.0


def bad_prefix_rate(hyps: List[str]) -> float:
    if not hyps:
        return 0.0
    bad = 0
    for h in hyps:
        low = h.strip().lower()
        if any(low.startswith(p) for p in BAD_PREFIXES):
            bad += 1
    return bad / len(hyps)


def corpus_metrics(srcs: List[str], refs: List[str], hyps: List[str]) -> Dict[str, float]:
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return {
        'bleu': float(bleu),
        'chrf': float(chrf),
        'number_match_rate': number_match_rate(srcs, hyps),
        'url_match_rate': url_match_rate(srcs, hyps),
        'bad_prefix_rate': bad_prefix_rate(hyps),
    }
