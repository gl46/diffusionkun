#!/usr/bin/env python
import argparse
from pathlib import Path

import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--model_prefix', required=True)
    ap.add_argument('--vocab_size', type=int, default=32000)
    args = ap.parse_args()

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    lang_tokens = [
        '<mask>',
        '<src_zh>', '<tgt_zh>', '<src_en>', '<tgt_en>',
        '<src_ja>', '<tgt_ja>', '<src_ko>', '<tgt_ko>',
        '<src_fr>', '<tgt_fr>', '<src_de>', '<tgt_de>', '<src_es>', '<tgt_es>',
    ]
    ph_tokens = [f'<PH{i}>' for i in range(100)]
    user_symbols = ','.join(lang_tokens + ph_tokens)

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=user_symbols,
        train_extremely_large_corpus=True,
    )


if __name__ == '__main__':
    main()
