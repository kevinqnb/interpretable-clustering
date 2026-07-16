#!/usr/bin/env bash
# Drives the full aniso experiment pipeline once per confidence threshold:
#   mine_rules (once, all thresholds)
#   -> per conf: [alphas || ids_lambda_search] -> select_alphas
#   -> lambda.py --emit-grid (once, all thresholds)
#   -> per conf: [max_rules || lambda]
#
# The pipeline runs as two per-confidence loops with a grid barrier between them, rather than one
# loop doing everything for a confidence before moving on. That split is forced by the shared
# lambda grid: it is anchored on every confidence's lambda*, and lambda* depends on that
# confidence's selected alpha -- so no grid can be built until select_alphas has run for ALL
# thresholds. See experiments/lambda_grid.py for why the grid is shared at all (short version:
# lambda* shrinks as confidence rises, so per-confidence grids ended at different lambdas and the
# high-confidence curves ran off the end of cross-confidence plots).
#
# Run from anywhere; paths below are relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIDENCE_THRESHOLDS=(0.25 0.5 0.75)

uv run python experiments/aniso/mine_rules.py \
    --confidence-thresholds "${CONFIDENCE_THRESHOLDS[@]}"

# Stage 1: per confidence, alphas.py and ids_lambda_search.py don't depend on each
# other -- run concurrently. ids_lambda_search is single-process, so alphas keeps
# its full default cpu-count. select_alphas.py then depends on alphas.py's output.
for CONF in "${CONFIDENCE_THRESHOLDS[@]}"; do
    echo "=== alphas: confidence=${CONF} ==="

    uv run python experiments/aniso/alphas.py --confidence "$CONF" --cpu-count 6 &
    ALPHAS_PID=$!
    uv run python experiments/aniso/ids_lambda_search.py --confidence "$CONF" &
    IDS_SEARCH_PID=$!
    wait "$ALPHAS_PID"
    wait "$IDS_SEARCH_PID"

    uv run python experiments/aniso/select_alphas.py --confidence "$CONF"
done

# Stage 2 (barrier): probe lambda* for every threshold and write the one shared lambda grid that
# every per-confidence lambda.py run below reads. Needs every selected_alphas file from stage 1.
echo "=== shared lambda grid (all thresholds) ==="
uv run python experiments/aniso/lambda.py --emit-grid \
    --confidence-thresholds "${CONFIDENCE_THRESHOLDS[@]}"

# Stage 3: per confidence, max_rules.py and lambda.py don't depend on each other, but
# both depend on stage 1 (select_alphas, and ids_lambda_search's cache); lambda.py also
# depends on stage 2's grid. Split the cpu budget between them since they run concurrently.
for CONF in "${CONFIDENCE_THRESHOLDS[@]}"; do
    echo "=== sweeps: confidence=${CONF} ==="

    uv run python experiments/aniso/max_rules.py --confidence "$CONF" --cpu-count 3 &
    MAX_RULES_PID=$!
    uv run python experiments/aniso/lambda.py --confidence "$CONF" --cpu-count 3 &
    LAMBDA_PID=$!
    wait "$MAX_RULES_PID"
    wait "$LAMBDA_PID"

    echo "=== done confidence=${CONF} ==="
done
