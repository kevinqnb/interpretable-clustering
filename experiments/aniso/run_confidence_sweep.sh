#!/usr/bin/env bash
# Drives the full aniso experiment pipeline once per confidence threshold:
#   mine_rules (once, all thresholds)
#   -> per conf: [alphas || ids_lambda_search] -> select_alphas
#   -> lambda.py --emit-grid (once, all thresholds)
#   -> per conf: [max_rules || lambda]
#
# The pipeline runs as two per-confidence loops with a grid barrier between them, rather than one
# loop doing everything for a confidence before moving on. That split is forced by the shared
# lambda grid: it is anchored on EVERY confidence's lambda*, so no grid can be built until the
# alpha stage has run. See experiments/lambda_grid.py for why the grid is shared at all (short
# version: lambda* shrinks as confidence rises, so per-confidence grids ended at different lambdas
# and the high-confidence curves ran off the end of cross-confidence plots).
#
# COMMON ALPHA (see ALPHA_CONFIDENCE below): stages 2 and 3 fit PEC against a single alpha shared
# by every threshold, rather than each threshold's own elbow-selected alpha.
#
# Run from anywhere; paths below are relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Must be ascending -- ALPHA_CONFIDENCE indexes this array positionally.
CONFIDENCE_THRESHOLDS=(0.25 0.5 0.75)

# The one threshold whose selected alphas every downstream fit uses.
#
# WHY: alpha is a parameter OF PEC's objective (reward - lambda*(cost + alpha*sum_rule_length)),
# not just of its solution. select_alphas.py picks it per confidence by elbow, and it genuinely
# moved -- on aniso's coverage-cost objective it was 284.58 at confidence 0.25/0.50 but 94.86 at
# 0.75. That made each confidence's plot score its models under a DIFFERENT objective function, so
# even models with nothing to do with the rule pool (ExKMC, CN2, Decision-Tree) drew different
# curves per confidence, purely from the alpha reweighting -- their fits are bit-identical across
# thresholds. Pinning one alpha makes the objective, and therefore the y-axis, mean the same thing
# in every subplot: the pool-independent baselines collapse to a single line, and PEC's curves stay
# comparable, still differing only through the rule pool the confidence filter actually changes.
#
# This is a REFIT, not a plotting change: PEC's selection depends on alpha, so scoring a
# per-confidence-alpha fit under a common alpha would be inconsistent. Hence the flag threads into
# the fits themselves.
#
# The middle-most threshold, taking the lower of the two on an even-length list: (n-1)/2 truncates.
ALPHA_CONFIDENCE=${CONFIDENCE_THRESHOLDS[$(((${#CONFIDENCE_THRESHOLDS[@]} - 1) / 2))]}
echo "=== common alpha source: confidence=${ALPHA_CONFIDENCE} ==="

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
# every per-confidence lambda.py run below reads. Needs stage 1's selected_alphas for
# ALPHA_CONFIDENCE (every threshold is probed against that one alpha); the per-threshold rule pools
# it also reads come from mine_rules. lambda* still varies by threshold, via the pool.
echo "=== shared lambda grid (all thresholds) ==="
uv run python experiments/aniso/lambda.py --emit-grid \
    --confidence-thresholds "${CONFIDENCE_THRESHOLDS[@]}" \
    --alpha-confidence "$ALPHA_CONFIDENCE"

# Stage 3: per confidence, max_rules.py and lambda.py don't depend on each other, but
# both depend on stage 1 (select_alphas, and ids_lambda_search's cache); lambda.py also
# depends on stage 2's grid. Split the cpu budget between them since they run concurrently.
for CONF in "${CONFIDENCE_THRESHOLDS[@]}"; do
    echo "=== sweeps: confidence=${CONF} ==="

    uv run python experiments/aniso/max_rules.py --confidence "$CONF" --cpu-count 3 \
        --alpha-confidence "$ALPHA_CONFIDENCE" &
    MAX_RULES_PID=$!
    uv run python experiments/aniso/lambda.py --confidence "$CONF" --cpu-count 3 \
        --alpha-confidence "$ALPHA_CONFIDENCE" &
    LAMBDA_PID=$!
    wait "$MAX_RULES_PID"
    wait "$LAMBDA_PID"

    echo "=== done confidence=${CONF} ==="
done
