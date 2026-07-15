#!/usr/bin/env bash
# Drives the full fashion experiment pipeline once per confidence threshold:
#   mine_rules (once, all thresholds)
#   -> [alphas || ids_lambda_search] -> select_alphas
#   -> [ (max_rules -> max_rules_exkmc -> max_rules_exp -> max_rules_combine)
#        || (lambda -> lambda_exkmc -> lambda_exp -> lambda_combine) ]
#
# fashion splits max_rules.py/lambda.py across companion _exkmc/_exp scripts (some
# algorithms take much longer to fit on this dataset -- see experiments/README.md step 3),
# then merges the three result files back together with *_combine.py. Each family runs as
# one background pipeline so the merge always happens after all three of its inputs exist;
# the two families run concurrently with each other since they don't depend on one another.
#
# Run from anywhere; paths below are relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIDENCE_THRESHOLDS=(0.25 0.5 0.75)

uv run python experiments/fashion/mine_rules.py \
    --confidence-thresholds "${CONFIDENCE_THRESHOLDS[@]}"

for CONF in "${CONFIDENCE_THRESHOLDS[@]}"; do
    echo "=== confidence=${CONF} ==="

    # Stage 1: alphas.py and ids_lambda_search.py don't depend on each
    # other -- run concurrently. ids_lambda_search is single-process, so
    # alphas keeps its full default cpu-count.
    uv run python experiments/fashion/alphas.py --confidence "$CONF" --cpu-count 12 &
    ALPHAS_PID=$!
    uv run python experiments/fashion/ids_lambda_search.py --confidence "$CONF" &
    IDS_SEARCH_PID=$!
    wait "$ALPHAS_PID"
    wait "$IDS_SEARCH_PID"

    # Stage 2: depends on alphas.py's output.
    uv run python experiments/fashion/select_alphas.py --confidence "$CONF"

    # Stage 3: the max_rules family and lambda family don't depend on each other, but both
    # depend on stage 2 (and ids_lambda_search's cache from stage 1) -- split the cpu budget
    # between the two heavy scripts (max_rules.py/lambda.py) since they run concurrently.
    # The companion _exkmc/_exp scripts are already cheap (cpu-count 1 by default) and run
    # sequentially within their own family, so they aren't split further.
    (
        uv run python experiments/fashion/max_rules.py --confidence "$CONF" --cpu-count 6
        uv run python experiments/fashion/max_rules_exkmc.py --confidence "$CONF" --cpu-count 1
        uv run python experiments/fashion/max_rules_exp.py --confidence "$CONF" --cpu-count 1
        uv run python experiments/fashion/max_rules_combine.py --confidence "$CONF"
    ) &
    MAX_RULES_PID=$!

    (
        uv run python experiments/fashion/lambda.py --confidence "$CONF" --cpu-count 6
        uv run python experiments/fashion/lambda_exkmc.py --confidence "$CONF" --cpu-count 1
        uv run python experiments/fashion/lambda_exp.py --confidence "$CONF" --cpu-count 1
        uv run python experiments/fashion/lambda_combine.py --confidence "$CONF"
    ) &
    LAMBDA_PID=$!

    wait "$MAX_RULES_PID"
    wait "$LAMBDA_PID"

    echo "=== done confidence=${CONF} ==="
done
