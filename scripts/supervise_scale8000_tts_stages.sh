#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNS_ROOT="${SLAIF_ASR_RUNS_ROOT:-/data/janezp/codex-work/slaif-asr-slovenian/runs}"
export SLAIF_ASR_RUNS_ROOT="$RUNS_ROOT"

CONFIG="${1:-configs/generation/scale8000_dual_gpu_generation_v1.json}"
RUN_DIR="$RUNS_ROOT/data-quality/sl-corpus-v5-scale8000-training-v1"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

count_wavs() {
  local path="$1"
  find "$path" -name '*.wav' 2>/dev/null | wc -l
}

stage_alive() {
  local pattern="$1"
  pgrep -f "$pattern" >/dev/null 2>&1
}

kill_stage() {
  local pattern="$1"
  local pid
  local pgid
  for pid in $(pgrep -f "$pattern" || true); do
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
    if [ -n "$pgid" ]; then
      kill -TERM "-$pgid" 2>/dev/null || true
    fi
  done
  sleep 10
  for pid in $(pgrep -f "$pattern" || true); do
    pgid="$(ps -o pgid= -p "$pid" | tr -d ' ' || true)"
    if [ -n "$pgid" ]; then
      kill -KILL "-$pgid" 2>/dev/null || true
    fi
  done
}

launch_piper() {
  echo "$(timestamp) launch_piper_dual"
  SCALE8000_PIPER_WORKERS_PER_GPU="${SCALE8000_PIPER_WORKERS_PER_GPU:-64}" \
    .venv/bin/python -u scripts/run_scale8000_dual_gpu_generation.py \
      --stage synthesize-piper-dual \
      --runs-root "$RUNS_ROOT" \
      --config "$CONFIG"
}

launch_supertonic() {
  echo "$(timestamp) launch_supertonic_dual"
  .venv/bin/python -u scripts/run_scale8000_dual_gpu_generation.py \
    --stage synthesize-supertonic-dual \
    --runs-root "$RUNS_ROOT" \
    --config "$CONFIG" \
    --progress-interval-seconds "${SCALE8000_SUPERTONIC_PROGRESS_SECONDS:-5}"
}

supervise_piper() {
  local last_total=-1
  local stale=0
  local gpu0
  local gpu1
  local total
  local alive

  while true; do
    gpu0="$(count_wavs "$RUN_DIR/piper-clean/gpu0/final-16000")"
    gpu1="$(count_wavs "$RUN_DIR/piper-clean/gpu1/final-16000")"
    total=$((gpu0 + gpu1))
    if stage_alive '[s]ynthesize-piper-worker'; then
      alive=yes
    else
      alive=no
    fi
    echo "$(timestamp) piper gpu0=$gpu0 gpu1=$gpu1 total=$total alive=$alive"

    if [ "$total" -ge 64000 ] && [ "$alive" = no ]; then
      echo "$(timestamp) piper_complete"
      break
    fi
    if [ "$total" -lt 64000 ] && [ "$alive" = no ]; then
      echo "$(timestamp) piper_restart total=$total"
      launch_piper
      stale=0
    elif [ "$total" = "$last_total" ]; then
      stale=$((stale + 1))
      if [ "$stale" -ge "${SCALE8000_STALE_MINUTES:-30}" ]; then
        echo "$(timestamp) piper_stale_restart total=$total"
        kill_stage '[s]ynthesize-piper-worker'
        launch_piper
        stale=0
      fi
    else
      stale=0
    fi
    last_total="$total"
    sleep "${SCALE8000_SUPERVISOR_INTERVAL_SECONDS:-60}"
  done

  .venv/bin/python -u scripts/run_scale8000_dual_gpu_generation.py \
    --stage merge-piper \
    --runs-root "$RUNS_ROOT" \
    --config "$CONFIG"
  echo "$(timestamp) piper_merged"
}

supervise_supertonic() {
  local last_total=-1
  local stale=0
  local gpu0
  local gpu1
  local total
  local alive

  while true; do
    gpu0="$(count_wavs "$RUN_DIR/supertonic-clean/gpu0/final-16000")"
    gpu1="$(count_wavs "$RUN_DIR/supertonic-clean/gpu1/final-16000")"
    total=$((gpu0 + gpu1))
    if stage_alive '[s]ynthesize-supertonic-worker'; then
      alive=yes
    else
      alive=no
    fi
    echo "$(timestamp) supertonic gpu0=$gpu0 gpu1=$gpu1 total=$total alive=$alive"

    if [ "$total" -ge 512000 ] && [ "$alive" = no ]; then
      echo "$(timestamp) supertonic_complete"
      break
    fi
    if [ "$total" -lt 512000 ] && [ "$alive" = no ]; then
      echo "$(timestamp) supertonic_restart total=$total"
      launch_supertonic
      stale=0
    elif [ "$total" = "$last_total" ]; then
      stale=$((stale + 1))
      if [ "$stale" -ge "${SCALE8000_STALE_MINUTES:-30}" ]; then
        echo "$(timestamp) supertonic_stale_restart total=$total"
        kill_stage '[s]ynthesize-supertonic-worker'
        launch_supertonic
        stale=0
      fi
    else
      stale=0
    fi
    last_total="$total"
    sleep "${SCALE8000_SUPERVISOR_INTERVAL_SECONDS:-60}"
  done

  .venv/bin/python -u scripts/run_scale8000_dual_gpu_generation.py \
    --stage merge-supertonic \
    --runs-root "$RUNS_ROOT" \
    --config "$CONFIG"
  echo "$(timestamp) supertonic_merged"
}

{
  echo "$(timestamp) supervisor_start config=$CONFIG runs_root=$RUNS_ROOT"
  supervise_piper
  supervise_supertonic
  echo "$(timestamp) supervisor_done"
} >>"$LOG_DIR/scale8000-supervisor.log" 2>&1
