#!/usr/bin/env bash
# Plot training results for an experiment.
# Usage: ./scripts/plot_experiment.sh <experiment_name> [run_name]
# Example: ./scripts/plot_experiment.sh salmon_hard_label
# Example: ./scripts/plot_experiment.sh salmon_hard_label run_2026_09_22_02_14
set -euo pipefail

EXPERIMENT_NAME="${1:-}"
RUN_NAME="${2:-}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "$EXPERIMENT_NAME" ]]; then
  echo "Usage: $0 <experiment_name> [run_name]"
  echo "Example: $0 salmon_hard_label"
  echo "Example: $0 salmon_hard_label run_2026_09_22_02_14"
  exit 1
fi

# Load .env so OUTPUT_DIR matches training launches.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.env"
  set +a
fi

if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
  echo "Using venv: ${PROJECT_ROOT}/.venv"
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Resolve experiment output directory:
# 1) configs/<name>.yaml General.output_dir (with ${OUTPUT_DIR} expanded)
# 2) ${OUTPUT_DIR:-./outputs}/<name>
BASE_DIR=""
CONFIG_FILE="${PROJECT_ROOT}/configs/${EXPERIMENT_NAME}.yaml"
if [[ -f "$CONFIG_FILE" ]]; then
  RAW_OUT=$(grep -E "^\s*output_dir:" "$CONFIG_FILE" | head -n 1 \
    | sed 's/.*output_dir:[[:space:]]*//' | tr -d '"' | tr -d "'" | xargs || true)
  if [[ -n "$RAW_OUT" ]]; then
    BASE_DIR=$(OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}" \
      python3 -c 'import os,sys; print(os.path.expandvars(sys.argv[1]))' "$RAW_OUT")
  fi
fi

if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
  BASE_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}/${EXPERIMENT_NAME}"
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: Experiment directory not found at $BASE_DIR"
  exit 1
fi

echo "Using output directory: $BASE_DIR"

# Pick a run directory.
if [[ -z "$RUN_NAME" ]]; then
  RUNS=$(ls -1d "$BASE_DIR"/run_* "$BASE_DIR"/test_run_* 2>/dev/null | sort -r || true)
  if [[ -z "$RUNS" ]]; then
    echo "Error: No run_* or test_run_* directories found in $BASE_DIR"
    exit 1
  fi

  echo "=========================================="
  echo "Runs for Experiment: $EXPERIMENT_NAME"
  echo "=========================================="
  echo ""

  i=1
  declare -a run_array
  while read -r run_path; do
    [[ -z "$run_path" ]] && continue
    if [[ ! -f "$run_path/config_resolved.yaml" ]]; then
      continue
    fi

    run_name=$(basename "$run_path")
    run_array[$i]="$run_path"

    has_train=false
    has_test=false
    [[ -f "$run_path/logs/metrics.jsonl" ]] && has_train=true
    if [[ -d "$run_path/logs" ]] && find "$run_path/logs" -maxdepth 1 -type d -name "test_*" 2>/dev/null | grep -q .; then
      has_test=true
    fi
    if [[ "$has_train" == true && "$has_test" == true ]]; then
      run_type="(train+test)"
    elif [[ "$has_train" == true ]]; then
      run_type="(train)"
    elif [[ "$has_test" == true ]]; then
      run_type="(test)"
    else
      run_type="(train)"
    fi

    log_count=0
    if [[ -f "$run_path/logs/metrics.jsonl" ]]; then
      log_count=$(wc -l < "$run_path/logs/metrics.jsonl" | tr -d ' ')
    fi
    if [[ "$log_count" -eq 0 && -d "$run_path/logs" ]]; then
      log_count=$(find "$run_path/logs" -maxdepth 2 -name "*.jsonl" -type f 2>/dev/null | wc -l | tr -d ' ')
    fi

    echo "[$i] $run_name $run_type ($log_count log entries)"
    i=$((i + 1))
  done <<< "$RUNS"

  if [[ "$i" -eq 1 ]]; then
    echo "Error: No run directories with config_resolved.yaml found in $BASE_DIR"
    exit 1
  fi

  echo ""
  read -r -p "Select run number (default: 1): " selection
  selection=${selection:-1}

  RUN_NAME="${run_array[$selection]:-}"
  if [[ -z "$RUN_NAME" ]]; then
    echo "Invalid selection"
    exit 1
  fi
else
  if [[ "$RUN_NAME" != /* ]]; then
    RUN_NAME="$BASE_DIR/$RUN_NAME"
  fi
fi

if [[ ! -d "$RUN_NAME" ]]; then
  echo "Error: Run directory not found at $RUN_NAME"
  exit 1
fi

echo "Plotting for experiment: $EXPERIMENT_NAME"
echo "Run directory: $RUN_NAME"

python3 -m src.analysis.plotter --experiment_dir "$RUN_NAME"

METRICS_FILE="$RUN_NAME/logs/metrics.jsonl"
if [[ ! -f "$METRICS_FILE" && -f "$RUN_NAME/metrics.jsonl" ]]; then
  METRICS_FILE="$RUN_NAME/metrics.jsonl"
fi

if [[ -f "$METRICS_FILE" ]]; then
  COMPONENT_OUT_DIR="$RUN_NAME/plots/metrics_components"
  echo ""
  echo "Generating component-loss plots from: $METRICS_FILE"
  if python3 -m src.analysis.plot_metrics_jsonl --metrics "$METRICS_FILE" --output_dir "$COMPONENT_OUT_DIR"; then
    echo "Saved component-loss plots to: $COMPONENT_OUT_DIR"
  else
    echo "Warning: Additional metrics.jsonl plotting failed."
  fi
else
  echo "No metrics.jsonl found for additional component-loss plotting."
fi

echo ""
echo "=========================================="
echo "Latest Model Generations (Preview)"
echo "=========================================="

LATEST_SAMPLES=$(ls -v "$RUN_NAME"/logs/samples_validation_epoch_*.jsonl 2>/dev/null | tail -n 1 || true)
if [[ -z "$LATEST_SAMPLES" ]]; then
  LATEST_SAMPLES=$(ls -v "$RUN_NAME"/logs/test_*/samples_test_epoch_*.jsonl 2>/dev/null | tail -n 1 || true)
fi
if [[ -z "$LATEST_SAMPLES" ]]; then
  LATEST_SAMPLES=$(ls -v "$RUN_NAME"/logs/test_*/predictions_test_epoch_*.jsonl 2>/dev/null | tail -n 1 || true)
fi

LATEST_JUDGE=""
if [[ -z "$LATEST_SAMPLES" ]]; then
  LATEST_JUDGE=$(ls -v "$RUN_NAME"/logs/judge_logs/epoch_*_iter_*.json 2>/dev/null | grep -v skeptic | tail -n 1 || true)
fi

if [[ -n "$LATEST_SAMPLES" ]]; then
  echo "Source: $LATEST_SAMPLES"
  echo ""
  head -n 5 "$LATEST_SAMPLES" | python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line)
    gt = str(data.get('gt', ''))
    out = str(data.get('output', ''))
    print(f'GT: {gt}')
    print(f'OUT: {out}')
    print('-' * 40)
"
elif [[ -n "$LATEST_JUDGE" ]]; then
  echo "Source (judge log): $LATEST_JUDGE"
  echo ""
  python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    entries = json.load(f)
for entry in entries[:2]:
    gt = str(entry.get('gt', ''))[:500]
    print(f\"audio_id: {entry.get('audio_id', 'n/a')}\")
    print(f'GT: {gt}')
    for i, gen in enumerate(entry.get('generations', [])[:2]):
        text = str(gen.get('text', ''))[:500]
        print(f'  gen[{i}] reward={gen.get(\"reward\")} format_ok={gen.get(\"format_ok\")} correct={gen.get(\"is_correct\")}')
        print(f'  OUT: {text}')
    print('-' * 40)
" "$LATEST_JUDGE"
else
  echo "No example generations found."
fi

PLOTS_DIR="$RUN_NAME/plots"
echo ""
echo "=========================================="
echo "Generated Plots"
echo "=========================================="
echo "Directory: $PLOTS_DIR"
echo "Full path: $(cd "$RUN_NAME" && pwd)/plots"
echo ""

list_plot() {
  local label="$1"
  local path="$2"
  if [[ -f "$path" ]]; then
    echo "  $label: $path"
  fi
}

list_plot "Training overview" "$PLOTS_DIR/training_overview.png"
list_plot "Training loss" "$PLOTS_DIR/training_loss.png"
list_plot "Validation loss" "$PLOTS_DIR/validation_loss.png"
list_plot "Validation accuracy" "$PLOTS_DIR/validation_accuracy.png"
list_plot "GRPO rewards" "$PLOTS_DIR/grpo_rewards_over_time.png"
list_plot "GRPO reward per epoch" "$PLOTS_DIR/grpo_reward_per_epoch.png"
list_plot "ASV / CM accuracy" "$PLOTS_DIR/validation_asv_cm_accuracy.png"
list_plot "ASV / CM EER" "$PLOTS_DIR/validation_asv_cm_eer.png"
list_plot "Sample generations" "$PLOTS_DIR/samples.html"

if [[ -d "$PLOTS_DIR/metrics_components" ]]; then
  list_plot "ASV / CM accuracy (components)" "$PLOTS_DIR/metrics_components/validation_asv_cm_accuracy.png"
  list_plot "ASV / CM EER (components)" "$PLOTS_DIR/metrics_components/validation_asv_cm_eer.png"
fi

if [[ -n "$LATEST_SAMPLES" && -f "$PLOTS_DIR/validation_asv_cm_accuracy.png" ]]; then
  echo ""
  echo "Latest SASV subsystem accuracies:"
  LATEST_SAMPLES="$LATEST_SAMPLES" PROJECT_ROOT="$PROJECT_ROOT" python3 <<'PYEOF' 2>/dev/null || true
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.environ["PROJECT_ROOT"])
from src.analysis.plotter_common import compute_sasv_subsystem_accuracies

samples_path = Path(os.environ["LATEST_SAMPLES"])
samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
accs = compute_sasv_subsystem_accuracies(samples)
if "asv_accuracy" in accs:
    print(f"  ASV accuracy (yes/no): {accs['asv_accuracy']:.4f}")
if "cm_accuracy" in accs:
    print(f"  CM accuracy (counter-measure): {accs['cm_accuracy']:.4f}")
PYEOF
fi

echo ""
echo "Done."
