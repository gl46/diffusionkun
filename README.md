# DiffusionKun starter

A scratch PyTorch prototype for a conditional masked-diffusion translator.

Default v0 target:

- direction: zh -> en
- model path: 30M debug -> 100M prototype -> 300M v0
- objective: source-conditioned masked denoising of target translation
- decoding: iterative confidence-based denoising, default 12 steps

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get raw data

Option A: use OPUS-100 from Hugging Face datasets:

```bash
python scripts/download_opus100.py \
  --pair en-zh \
  --src_lang zh \
  --tgt_lang en \
  --split train \
  --output data/raw/opus100_zh_en.raw.jsonl \
  --limit 1000000 \
  --streaming
```

Raw format expected by cleaning script:

```jsonl
{"src":"中文源句","tgt":"English target sentence."}
```

## 3. Clean and split

```bash
python scripts/clean_parallel.py \
  --input data/raw/opus100_zh_en.raw.jsonl \
  --output data/cleaned/zh_en.all.clean.jsonl \
  --reject data/rejected/zh_en.rejected.jsonl

python scripts/split_jsonl.py \
  --input data/cleaned/zh_en.all.clean.jsonl \
  --train data/cleaned/zh_en.train.jsonl \
  --dev data/cleaned/zh_en.dev.jsonl \
  --dev_size 5000
```

For overfitting/debug:

```bash
python scripts/sample_debug_set.py \
  --input data/cleaned/zh_en.train.jsonl \
  --output data/cleaned/zh_en.train.debug1000.jsonl \
  --n 1000
```

Then edit `configs/30m_debug.yaml` and point `train_jsonl` to `data/cleaned/zh_en.train.debug1000.jsonl` for the first sanity check.

## 4. Train SentencePiece

```bash
python scripts/make_spm_corpus.py \
  --input data/cleaned/zh_en.train.jsonl \
  --output data/tokenized/spm_train.txt \
  --limit 1000000

python scripts/train_sentencepiece.py \
  --input data/tokenized/spm_train.txt \
  --model_prefix data/tokenized/diffusionkun_zh_en_32k \
  --vocab_size 32000
```

## 5. Train 30M debug

```bash
python train.py --config configs/30m_debug.yaml
```

When this can overfit 1000 examples, move to:

```bash
python train.py --config configs/100m_zh_en.yaml
python train.py --config configs/300m_zh_en.yaml
```

## 6. Decode

```bash
python decode.py \
  --checkpoint checkpoints/30m_debug/final.pt \
  --text "这个模型主要瓶颈不是算力，而是解码方式。" \
  --steps 12
```

## 7. Evaluate

Oracle length first:

```bash
python eval.py \
  --checkpoint checkpoints/30m_debug/final.pt \
  --eval_jsonl data/cleaned/zh_en.dev.jsonl \
  --oracle_length \
  --limit 1000
```

Predicted length:

```bash
python eval.py \
  --checkpoint checkpoints/30m_debug/final.pt \
  --eval_jsonl data/cleaned/zh_en.dev.jsonl \
  --limit 1000
```

If oracle length works but predicted length is bad, debug the length head. If both are bad on a tiny overfit set, debug model/loss/sampling.

## Notes

- The decoder uses bidirectional target self-attention. Do not add a causal target mask.
- The source is never noised; only the target canvas is corrupted.
- v0 uses masked diffusion/CMLM-style denoising, not a full uniform-state diffusion recipe.
- Keep the first v0 narrow: zh -> en, max 128 source tokens, max 128 target tokens.
