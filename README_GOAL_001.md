# Goal-001: Qwen3 Embedding Cleaning + Long Run

Run from the repo root on the A100 node. Do not run the optional Qwen3.6-27B
audit server on the same GPU while BabyDiffMT is training unless you explicitly
want both jobs to share memory.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Full Cleaning + 30M + 100M

```bash
LIMIT=5000000 \
HF_ENDPOINT=https://hf-mirror.com \
EMB_MODEL=Qwen/Qwen3-Embedding-0.6B \
EMB_BATCH=256 \
EMB_THRESHOLD_STRATEGY=balanced \
CALIBRATION_SIZE=50000 \
RUN_100M=1 \
RUN_AUDIT=0 \
bash scripts/run_goal_001.sh 2>&1 | tee logs/goal_001_$(date +%F_%H%M).log
```

`EMB_THRESHOLD_STRATEGY=balanced` records the strict
`max(negative_p99 + 0.02, positive_p05)` candidate in the calibration report,
but uses `positive_p05` as the reject line. This avoids over-rejecting short
titles, organization names, and template-like OPUS rows. Set
`EMB_THRESHOLD_STRATEGY=strict` to use the literal strict candidate.

The script writes:

- `data/raw/opus100_zh_en.raw.jsonl`
- `data/cleaned/zh_en.rule.clean.jsonl`
- `data/rejected/zh_en.rule.rejected.jsonl`
- `runs/goal_001/reject_rule_report.txt`
- `runs/goal_001/qwen3_embedding_calibration.json`
- `runs/goal_001/qwen3_embedding_calibration.txt`
- `data/cleaned/zh_en.embed.clean.jsonl`
- `data/rejected/zh_en.embed.rejected.jsonl`
- `runs/goal_001/embed_quality_report.txt`
- `runs/goal_001/eval_30m_oracle.txt`
- `runs/goal_001/eval_30m_top1.txt`
- `runs/goal_001/eval_30m_top3.txt`
- `runs/goal_001/eval_100m_oracle.txt`
- `runs/goal_001/eval_100m_top3.txt`
- `runs/goal_001/summary.txt`

## Cleaning / Filtering Only

```bash
RUN_100M=0 STEPS_30M=1 LIMIT=5000000 bash scripts/run_goal_001.sh
```

For direct embedding filtering:

```bash
python3 scripts/embedding_filter.py \
  --input data/cleaned/zh_en.rule.clean.jsonl \
  --output data/cleaned/zh_en.embed.clean.jsonl \
  --reject data/rejected/zh_en.embed.rejected.jsonl \
  --report runs/goal_001/embed_quality_report.txt \
  --model Qwen/Qwen3-Embedding-0.6B \
  --auto_thresholds \
  --threshold_strategy balanced \
  --calibration_size 50000 \
  --batch_size 256
```

If the node can reach `huggingface.co` directly, omit `HF_ENDPOINT`. On node2,
use `HF_ENDPOINT=https://hf-mirror.com`.

## Optional Qwen3.6-27B Audit

In a separate environment/session:

```bash
pip install vllm
bash scripts/launch_qwen36_27b_vllm.sh
```

Then, when no BabyDiffMT training job is using the same GPU:

```bash
python3 scripts/llm_audit_qwen.py \
  --input data/cleaned/zh_en.embed.clean.jsonl \
  --input data/rejected/zh_en.embed.rejected.jsonl \
  --output runs/goal_001/llm_audit_boundary.jsonl \
  --thresholds_json runs/goal_001/qwen3_embedding_calibration.json \
  --n 200
```

## Show Reports

```bash
python3 scripts/summarize_goal_001.py --run_dir runs/goal_001
```
