#!/usr/bin/env bash
# Rerun few-shot conditions for the 3 closed models (GPT-5-mini, GPT-5.2, Claude-Haiku-4.5)
# against the 2 new exemplar seeds (101, 202), matching the open-model seed ablation
# already completed on HPC (data_out/few_shot_seed101/, data_out/few_shot_seed202/).
#
# Written for bash 3.2 (macOS default) -- no associative arrays, no `timeout` binary.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a

SHOTS="3 6 9 12 15"
SEEDS="101 202"
DATASETS="semeval2016_task6_testdata_gold ezstance_test mtcsd_test"
RUN_TIMEOUT_SECS=2400   # 40 min hard cap per run; kill and move on if exceeded

# Optional first arg: "openai", "claude", or omit/"all" to run both (default).
# Lets the two backends run as separate concurrent background jobs -- they hit
# different APIs with independent rate limits and write to different folders,
# so there's no conflict running them side by side.
BACKEND_FILTER="${1:-all}"
case "$BACKEND_FILTER" in
  openai) MODEL_ENTRIES="openai:gpt-5-mini openai:gpt-5.2" ;;
  claude) MODEL_ENTRIES="claude:claude-haiku-4-5-20251001" ;;
  all) MODEL_ENTRIES="openai:gpt-5-mini openai:gpt-5.2 claude:claude-haiku-4-5-20251001" ;;
  *) echo "Unknown backend filter: $BACKEND_FILTER (expected openai/claude/all)"; exit 1 ;;
esac

dataset_path() {
  case "$1" in
    semeval2016_task6_testdata_gold) echo "data_in/semeval/semeval2016_task6_testdata_gold" ;;
    ezstance_test) echo "data_in/ezstance/ezstance_test.csv" ;;
    mtcsd_test) echo "data_in/mtcsd/mtcsd_test.csv" ;;
  esac
}

pool_prefix() {
  case "$1" in
    semeval2016_task6_testdata_gold) echo "semeval" ;;
    ezstance_test) echo "ezstance" ;;
    mtcsd_test) echo "mtcsd" ;;
  esac
}

# Pure-bash timeout wrapper (no `timeout`/`gtimeout` binary available on this machine).
# Uses wall-clock timestamps (not a loop counter) so it self-corrects even if the
# machine sleeps mid-run -- a naive counter under-counts time spent suspended and
# lets the process run far past the intended cap once the machine wakes up.
run_with_timeout() {
  local limit="$1"
  shift
  "$@" &
  local cmd_pid=$!
  local start_ts
  start_ts=$(date +%s)
  while kill -0 "$cmd_pid" 2>/dev/null; do
    local now_ts elapsed
    now_ts=$(date +%s)
    elapsed=$((now_ts - start_ts))
    if [ "$elapsed" -ge "$limit" ]; then
      echo "!!! TIMEOUT after ${elapsed}s (limit ${limit}s) -- killing PID $cmd_pid and moving on"
      kill -TERM "$cmd_pid" 2>/dev/null
      sleep 3
      kill -KILL "$cmd_pid" 2>/dev/null
      wait "$cmd_pid" 2>/dev/null
      return 124
    fi
    sleep 10
  done
  wait "$cmd_pid"
  return $?
}

RUN=0
n_models=$(echo $MODEL_ENTRIES | wc -w | tr -d ' ')
n_shots=$(echo $SHOTS | wc -w | tr -d ' ')
n_seeds=$(echo $SEEDS | wc -w | tr -d ' ')
n_datasets=$(echo $DATASETS | wc -w | tr -d ' ')
TOTAL=$((n_models * n_shots * n_seeds * n_datasets))
echo "Backend filter: $BACKEND_FILTER ($TOTAL total runs for this filter)"
FAILED=""

for seed in $SEEDS; do
  for ds in $DATASETS; do
    pool="data_in/few_shot_seeds/$(pool_prefix "$ds")_master_seed${seed}.json"
    for model_entry in $MODEL_ENTRIES; do
      backend="${model_entry%%:*}"
      model="${model_entry#*:}"
      for shot in $SHOTS; do
        RUN=$((RUN+1))
        expected_dir="data_out/few_shot_seed${seed}/${ds}/GenAIStanceOneShot_"*"_${ds}_${model}_equal_few_shot_${shot}"
        if compgen -G "$expected_dir/document_assignments.csv" > /dev/null; then
          echo "=== [$RUN/$TOTAL] SKIP (already done) seed=$seed ds=$ds model=$model shot=$shot ==="
          continue
        fi
        echo "=== [$RUN/$TOTAL] seed=$seed ds=$ds backend=$backend model=$model shot=$shot ==="
        cmd=(python RunModels.py --dataset GENERIC
          --set "CUSTOM_DATASET_PATH=$(dataset_path "$ds")"
          --set "CATEGORY_COLUMN=stance_label"
          --set "USE_ALL_DOCUMENTS=true"
          --set "LLM_BACKEND=${backend}"
          --set "FEW_SHOT_MASTER_PATH=${pool}"
          --task-type stance_detection
          --method-type GenAIStanceOneShot
          --prompt-name "few_shot_${shot}"
          --query-column query
          --output-dir "data_out/few_shot_seed${seed}")
        if [ "$backend" = "openai" ]; then
          cmd+=(--set "OPENAI_MODEL=${model}")
        else
          cmd+=(--set "CLAUDE_MODEL=${model}" --set "CLAUDE_BATCH_SIZE=25")
        fi
        echo "${cmd[@]}"
        run_with_timeout "$RUN_TIMEOUT_SECS" "${cmd[@]}"
        status=$?
        if [ "$status" -ne 0 ]; then
          echo "!!! RUN FAILED (exit $status): seed=$seed ds=$ds model=$model shot=$shot"
          FAILED="${FAILED}\n  seed=$seed ds=$ds model=$model shot=$shot (exit $status)"
        fi
      done
    done
  done
done

echo "ALL DONE: $TOTAL runs processed"
if [ -n "$FAILED" ]; then
  echo "FAILURES:"
  printf "%b\n" "$FAILED"
fi
