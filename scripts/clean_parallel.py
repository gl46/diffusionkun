#!/usr/bin/env python
import argparse
import json
import re
import unicodedata
from pathlib import Path

ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://[^\s]+")
NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BAD_PREFIX_RE = re.compile(r"^(译文如下|这句话可以翻译为|the translation is|here is the translation|translation:)\s*", re.I)


def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\u200b', '').replace('\ufeff', '')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def zh_ratio(s: str) -> float:
    return len(ZH_RE.findall(s)) / max(len(s), 1)


def en_ratio(s: str) -> float:
    return len(EN_RE.findall(s)) / max(len(s), 1)


def nums(s: str):
    return sorted(x.replace(',', '') for x in NUM_RE.findall(s))


def urls(s: str):
    return sorted(URL_RE.findall(s))


def too_repetitive(s: str) -> bool:
    if len(s) < 12:
        return False
    return max(s.count(ch) for ch in set(s)) / len(s) > 0.5


def has_html_garbage(s: str) -> bool:
    if HTML_TAG_RE.search(s) or HTML_ENTITY_RE.search(s):
        return True
    if CONTROL_RE.search(s) or '\ufffd' in s:
        return True
    if len(s) >= 20:
        textish = len(re.findall(r"[\w\s\u4e00-\u9fff.,!?;:'\"()\[\]{}，。！？；：“”‘’、\-]", s))
        if textish / max(len(s), 1) < 0.55:
            return True
    return False


def rule_filter(src: str, tgt: str):
    if not src or not tgt:
        return False, 'empty'
    if src == tgt:
        return False, 'same_src_tgt'
    if len(src) < 2 or len(tgt) < 2:
        return False, 'too_short'
    if len(src) > 2000 or len(tgt) > 4000:
        return False, 'too_long'
    ratio = len(tgt) / max(len(src), 1)
    if ratio < 0.25 or ratio > 6.0:
        return False, 'bad_length_ratio'
    if zh_ratio(src) < 0.12:
        return False, 'src_not_zh'
    if en_ratio(tgt) < 0.25:
        return False, 'tgt_not_en'
    if has_html_garbage(src) or has_html_garbage(tgt):
        return False, 'html_or_garbage'
    if too_repetitive(src) or too_repetitive(tgt):
        return False, 'repetitive'
    if urls(src) != urls(tgt):
        return False, 'url_mismatch'
    # Relaxed for zh-en: only compare ASCII numbers when both sides contain ASCII digits.
    # This avoids rejecting good pairs like "三个人" -> "three people" or dates written in Chinese numerals.
    src_nums = nums(src)
    tgt_nums = nums(tgt)
    if src_nums and tgt_nums and src_nums != tgt_nums:
        return False, 'number_mismatch'
    if BAD_PREFIX_RE.match(tgt):
        return False, 'assistant_prefix'
    return True, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='raw jsonl with src/tgt fields')
    ap.add_argument('--output', required=True)
    ap.add_argument('--reject', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.reject).parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = rejected = 0
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout, \
         open(args.reject, 'w', encoding='utf-8') as frej:
        for i, line in enumerate(fin):
            if args.limit and i >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            src = normalize_text(obj.get('src', ''))
            tgt = normalize_text(obj.get('tgt', ''))
            key = (src, tgt)
            if key in seen:
                obj['reject_reason'] = 'duplicate'
                frej.write(json.dumps(obj, ensure_ascii=False) + '\n')
                rejected += 1
                continue
            seen.add(key)
            ok, reason = rule_filter(src, tgt)
            if ok:
                fout.write(json.dumps({'src_lang': 'zh', 'tgt_lang': 'en', 'src': src, 'tgt': tgt}, ensure_ascii=False) + '\n')
                kept += 1
            else:
                obj['src'] = src
                obj['tgt'] = tgt
                obj['reject_reason'] = reason
                frej.write(json.dumps(obj, ensure_ascii=False) + '\n')
                rejected += 1
    print({'kept': kept, 'rejected': rejected})


if __name__ == '__main__':
    main()
