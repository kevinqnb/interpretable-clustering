#!/usr/bin/env bash
# Drives the full anuran experiment pipeline once per confidence threshold:
#   mine_rules (once, all thresholds)
#   -> [alphas || ids_lambda_search] -> select_alphas -> [max_rules || lambda]
#
# Run from anywhere; paths below are relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIDENCE_THRESHOLDS=(0.25 0.5 0.75)

uv run python experiments/anuran/mine_rules.py \
    --confidence-thresholds "${CONFIDENCE_THRESHOLDS[@]}"

for CONF in "${CONFIDENCE_THRESHOLDS[@]}"; do
    echo "=== confidence=${CONF} ==="

    # Stage 1: alphas.py and ids_lambda_search.py don't depend on each
    # other -- run concurrently. ids_lambda_search is single-process, so
    # alphas keeps its full default cpu-count.
    uv run python experiments/anuran/alphas.py --confidence "$CONF" --cpu-count 12 &
    ALPHAS_PID=$!
    uv run python experiments/anuran/ids_lambda_search.py --confidence "$CONF" &
    IDS_SEARCH_PID=$!
    wait "$ALPHAS_PID"
    wait "$IDS_SEARCH_PID"

    # Stage 2: depends on alphas.py's output.
    uv run python experiments/anuran/select_alphas.py --confidence "$CONF"

    # Stage 3: max_rules.py and lambda.py don't depend on each other, but
    # both depend on stage 2 (and ids_lambda_search's cache from stage 1) --
    # split the cpu budget between them since they run concurrently.
    uv run python experiments/anuran/max_rules.py --confidence "$CONF" --cpu-count 1 &
    MAX_RULES_PID=$!
    uv run python experiments/anuran/lambda.py --confidence "$CONF" --cpu-count 1 &
    LAMBDA_PID=$!
    wait "$MAX_RULES_PID"
    wait "$LAMBDA_PID"

    echo "=== done confidence=${CONF} ==="
done
