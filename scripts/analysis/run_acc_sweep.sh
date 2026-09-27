#!/usr/bin/env bash
# Driver for the multi-round accumulation study (exp_floor_matcher.py specs
# `loftr:acc=...`).  All specs go to ONE python invocation so the GPU is
# acquired once (the card is shared); exp_floor_matcher writes each
# (scene, spec) JSON as it finishes, so a kill only loses the spec in flight.
# Waits for free VRAM before starting and retries the batch on failure.
# $ACC_GPU_FREE_MB (default 4000) is the gate; the run itself peaks at ~1.25 GB,
# and every frame retries on a transient OOM, so a lower gate is safe when the
# shared card is parked just under 4000 MB by other jobs.
#
#   scripts/analysis/run_acc_sweep.sh <outdir> "<scenes>" "<spec> [<spec>...]" [extra args]
set -u
cd "$(dirname "$0")/../.."
OUT="$1"; SCENES="$2"; SPECS="$3"; shift 3
GATE=${ACC_GPU_FREE_MB:-4000}
mkdir -p "$OUT"
for attempt in 1 2 3 4; do
  for _ in $(seq 480); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$GATE" ] && break
    sleep 15
  done
  echo "=== batch attempt $attempt -> $OUT :: $SPECS"
  python3 scripts/analysis/exp_floor_matcher.py $SCENES --matchers $SPECS \
      --out "$OUT" "$@" 2>&1 | tee -a "$OUT/run.log"
  st=${PIPESTATUS[0]}
  [ "$st" -eq 0 ] && break
  echo "  batch failed (exit $st); retrying in 120 s"; sleep 120
done
