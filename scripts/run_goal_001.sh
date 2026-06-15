#!/usr/bin/env bash
set -euo pipefail

# Goal-001 pipeline:
# rule clean -> Qwen3 embedding calibration/filter -> tokenizer -> 30M debug
# -> optional 100M prototype -> summary.

LIMIT=${LIMIT:-5000000}
EMB_MODEL=${EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B}
EMB_BATCH=${EMB_BATCH:-256}
EMB_DEVICE=${EMB_DEVICE:-cuda}
EMB_PRECISION=${EMB_PRECISION:-auto}
EMB_THRESHOLD_STRATEGY=${EMB_THRESHOLD_STRATEGY:-balanced}
RUN_AUDIT=${RUN_AUDIT:-0}
RUN_100M=${RUN_100M:-1}
CALIBRATION_SIZE=${CALIBRATION_SIZE:-50000}
PYTHON=${PYTHON:-python3}

DEV_SIZE=${DEV_SIZE:-5000}
SPM_VOCAB_SIZE=${SPM_VOCAB_SIZE:-32000}
SPM_CORPUS_LIMIT=${SPM_CORPUS_LIMIT:-1000000}
STEPS_30M=${STEPS_30M:-5000}
STEPS_100M=${STEPS_100M:-30000}
EVAL_LIMIT_30M=${EVAL_LIMIT_30M:-1000}
EVAL_LIMIT_100M=${EVAL_LIMIT_100M:-2000}
DECODE_STEPS=${DECODE_STEPS:-12}

RUN_DIR=${RUN_DIR:-runs/goal_001}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$RUN_DIR" "$RUN_DIR/configs" "$LOG_DIR" data/raw data/cleaned data/rejected data/tokenized checkpoints

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi

RAW=data/raw/opus100_zh_en.raw.jsonl
RULE_CLEAN=data/cleaned/zh_en.rule.clean.jsonl
RULE_REJECT=data/rejected/zh_en.rule.rejected.jsonl
EMBED_CLEAN=data/cleaned/zh_en.embed.clean.jsonl
EMBED_REJECT=data/rejected/zh_en.embed.rejected.jsonl
TRAIN_JSONL=data/cleaned/zh_en.train.jsonl
DEV_JSONL=data/cleaned/zh_en.dev.jsonl
DEBUG_JSONL=data/cleaned/zh_en.train.debug1000.jsonl
SPM_PREFIX=data/tokenized/diffusionkun_zh_en_32k
SPM_MODEL=${SPM_PREFIX}.model

echo "[1/13] download OPUS-100 zh->en streaming sample limit=$LIMIT"
if [[ "${FORCE_DOWNLOAD:-0}" == "1" || ! -s "$RAW" ]]; then
  "$PYTHON" scripts/download_opus100.py \
    --pair en-zh \
    --src_lang zh \
    --tgt_lang en \
    --split train \
    --output "$RAW" \
    --limit "$LIMIT" \
    --streaming 2>&1 | tee "$RUN_DIR/download_opus100.log"
else
  echo "skip download; existing $RAW"
fi

echo "[2/13] rule clean"
"$PYTHON" scripts/clean_parallel.py \
  --input "$RAW" \
  --output "$RULE_CLEAN" \
  --reject "$RULE_REJECT" 2>&1 | tee "$RUN_DIR/rule_clean.log"

{
  echo "[rule clean kept]"
  "$PYTHON" scripts/report_jsonl_counts.py --input "$RULE_CLEAN" --field src_lang
  echo
  echo "[rule clean rejected]"
  "$PYTHON" scripts/report_jsonl_counts.py --input "$RULE_REJECT" --field reject_reason
} | tee "$RUN_DIR/reject_rule_report.txt"

echo "[3/13] Qwen3 embedding calibration and semantic filter model=$EMB_MODEL"
"$PYTHON" scripts/embedding_filter.py \
  --input "$RULE_CLEAN" \
  --output "$EMBED_CLEAN" \
  --reject "$EMBED_REJECT" \
  --report "$RUN_DIR/embed_quality_report.txt" \
  --model "$EMB_MODEL" \
  --batch_size "$EMB_BATCH" \
  --device "$EMB_DEVICE" \
  --precision "$EMB_PRECISION" \
  --auto_thresholds \
  --threshold_strategy "$EMB_THRESHOLD_STRATEGY" \
  --calibration_size "$CALIBRATION_SIZE" \
  --calibration_json "$RUN_DIR/qwen3_embedding_calibration.json" \
  --calibration_txt "$RUN_DIR/qwen3_embedding_calibration.txt" 2>&1 | tee "$RUN_DIR/embedding_filter.log"

{
  echo "[embedding kept buckets]"
  "$PYTHON" scripts/report_jsonl_counts.py --input "$EMBED_CLEAN" --field emb_bucket
  echo
  echo "[embedding rejected reasons]"
  "$PYTHON" scripts/report_jsonl_counts.py --input "$EMBED_REJECT" --field reject_reason
} | tee "$RUN_DIR/reject_embed_report.txt"

echo "[4/13] optional Qwen3.6-27B LLM audit"
if [[ "$RUN_AUDIT" == "1" ]]; then
  echo "audit assumes a local qwen-audit OpenAI-compatible server at http://127.0.0.1:8001/v1"
  "$PYTHON" scripts/llm_audit_qwen.py \
    --input "$EMBED_CLEAN" \
    --input "$EMBED_REJECT" \
    --output "$RUN_DIR/llm_audit_boundary.jsonl" \
    --n "${AUDIT_N:-200}" \
    --thresholds_json "$RUN_DIR/qwen3_embedding_calibration.json" \
    --base_url "${AUDIT_BASE_URL:-http://127.0.0.1:8001/v1}" \
    --model "${AUDIT_MODEL:-qwen-audit}" 2>&1 | tee "$RUN_DIR/llm_audit.log"
else
  echo "skip audit; set RUN_AUDIT=1 only when qwen-audit server is running and GPU memory is free"
fi

echo "[5/13] split train/dev"
"$PYTHON" scripts/split_jsonl.py \
  --input "$EMBED_CLEAN" \
  --train "$TRAIN_JSONL" \
  --dev "$DEV_JSONL" \
  --dev_size "$DEV_SIZE" 2>&1 | tee "$RUN_DIR/split.log"

echo "[6/13] train SentencePiece tokenizer if missing"
if [[ "${FORCE_SPM:-0}" == "1" || ! -s "$SPM_MODEL" ]]; then
  "$PYTHON" scripts/make_spm_corpus.py \
    --input "$TRAIN_JSONL" \
    --output data/tokenized/spm_train.txt \
    --limit "$SPM_CORPUS_LIMIT" 2>&1 | tee "$RUN_DIR/make_spm_corpus.log"
  "$PYTHON" scripts/train_sentencepiece.py \
    --input data/tokenized/spm_train.txt \
    --model_prefix "$SPM_PREFIX" \
    --vocab_size "$SPM_VOCAB_SIZE" 2>&1 | tee "$RUN_DIR/train_sentencepiece.log"
else
  echo "skip SentencePiece; existing $SPM_MODEL"
fi

echo "[7/13] sample debug1000"
"$PYTHON" scripts/sample_debug_set.py \
  --input "$TRAIN_JSONL" \
  --output "$DEBUG_JSONL" \
  --n 1000 2>&1 | tee "$RUN_DIR/sample_debug1000.log"

echo "[8/13] write Goal-001 generated configs"
RUN_DIR="$RUN_DIR" STEPS_30M="$STEPS_30M" STEPS_100M="$STEPS_100M" SPM_MODEL="$SPM_MODEL" \
TRAIN_JSONL="$TRAIN_JSONL" DEV_JSONL="$DEV_JSONL" DEBUG_JSONL="$DEBUG_JSONL" "$PYTHON" - <<'PY'
import os
from pathlib import Path

import yaml

run_dir = Path(os.environ['RUN_DIR'])
spm_model = os.environ['SPM_MODEL']
train_jsonl = os.environ['TRAIN_JSONL']
dev_jsonl = os.environ['DEV_JSONL']
debug_jsonl = os.environ['DEBUG_JSONL']

cfg30 = yaml.safe_load(open('configs/30m_debug.yaml', encoding='utf-8'))
cfg30['paths']['train_jsonl'] = debug_jsonl
cfg30['paths']['dev_jsonl'] = debug_jsonl
cfg30['paths']['spm_model'] = spm_model
cfg30['paths']['output_dir'] = 'checkpoints/30m_goal001'
cfg30['training']['max_steps'] = int(os.environ['STEPS_30M'])
cfg30['training']['eval_every'] = min(int(cfg30['training']['eval_every']), max(100, int(os.environ['STEPS_30M']) // 5))
cfg30['training']['save_every'] = min(int(cfg30['training']['save_every']), max(100, int(os.environ['STEPS_30M']) // 2))
(run_dir / 'configs/30m_goal001.yaml').write_text(
    yaml.safe_dump(cfg30, allow_unicode=True, sort_keys=False),
    encoding='utf-8',
)

cfg100 = yaml.safe_load(open('configs/100m_zh_en.yaml', encoding='utf-8'))
cfg100['paths']['train_jsonl'] = train_jsonl
cfg100['paths']['dev_jsonl'] = dev_jsonl
cfg100['paths']['spm_model'] = spm_model
cfg100['paths']['output_dir'] = 'checkpoints/100m_goal001'
cfg100['training']['max_steps'] = int(os.environ['STEPS_100M'])
(run_dir / 'configs/100m_goal001.yaml').write_text(
    yaml.safe_dump(cfg100, allow_unicode=True, sort_keys=False),
    encoding='utf-8',
)
PY

echo "[9/13] train 30M debug"
"$PYTHON" train.py --config "$RUN_DIR/configs/30m_goal001.yaml" 2>&1 | tee "$RUN_DIR/train_30m_debug.log"

echo "[10/13] evaluate 30M debug"
"$PYTHON" eval.py \
  --checkpoint checkpoints/30m_goal001/final.pt \
  --oracle_length \
  --steps "$DECODE_STEPS" \
  --limit "$EVAL_LIMIT_30M" \
  --output "$RUN_DIR/30m_oracle.jsonl" 2>&1 | tee "$RUN_DIR/eval_30m_oracle.txt"

"$PYTHON" eval.py \
  --checkpoint checkpoints/30m_goal001/final.pt \
  --steps "$DECODE_STEPS" \
  --length_topk 1 \
  --limit "$EVAL_LIMIT_30M" \
  --output "$RUN_DIR/30m_top1.jsonl" 2>&1 | tee "$RUN_DIR/eval_30m_top1.txt"

"$PYTHON" eval.py \
  --checkpoint checkpoints/30m_goal001/final.pt \
  --steps "$DECODE_STEPS" \
  --length_topk 3 \
  --limit "$EVAL_LIMIT_30M" \
  --output "$RUN_DIR/30m_top3.jsonl" 2>&1 | tee "$RUN_DIR/eval_30m_top3.txt"

echo "[11/13] optional 100M prototype"
if [[ "$RUN_100M" == "1" ]]; then
  "$PYTHON" train.py --config "$RUN_DIR/configs/100m_goal001.yaml" 2>&1 | tee "$RUN_DIR/train_100m.log"

  echo "[12/13] evaluate 100M prototype"
  "$PYTHON" eval.py \
    --checkpoint checkpoints/100m_goal001/final.pt \
    --oracle_length \
    --steps "$DECODE_STEPS" \
    --limit "$EVAL_LIMIT_100M" \
    --output "$RUN_DIR/100m_oracle.jsonl" 2>&1 | tee "$RUN_DIR/eval_100m_oracle.txt"

  "$PYTHON" eval.py \
    --checkpoint checkpoints/100m_goal001/final.pt \
    --steps "$DECODE_STEPS" \
    --length_topk 3 \
    --limit "$EVAL_LIMIT_100M" \
    --output "$RUN_DIR/100m_top3.jsonl" 2>&1 | tee "$RUN_DIR/eval_100m_top3.txt"
else
  echo "skip 100M; set RUN_100M=1 to train the bounded prototype"
fi

echo "[13/13] summarize"
"$PYTHON" scripts/summarize_goal_001.py --run_dir "$RUN_DIR" --samples 10 2>&1 | tee "$RUN_DIR/summary.txt"

if grep -q '<mask>' "$RUN_DIR/30m_top3.jsonl" 2>/dev/null; then
  echo "warning: residual <mask> found in 30M top-3 output" | tee -a "$RUN_DIR/summary.txt"
fi
if [[ "$RUN_100M" == "1" ]] && grep -q '<mask>' "$RUN_DIR/100m_top3.jsonl" 2>/dev/null; then
  echo "warning: residual <mask> found in 100M top-3 output" | tee -a "$RUN_DIR/summary.txt"
fi

echo "Goal-001 finished. Reports are under $RUN_DIR; top-level command logs can be tee'd into $LOG_DIR."
