#!/usr/bin/env bash
# Alternating training runs: vanilla and tiered, sharing a single GPU.
# First iteration: vanilla 30 gens, then alternates 20 gens each.
#
# Usage:
#   ./scripts/alternating_runs.sh          # Run indefinitely
#   ./scripts/alternating_runs.sh 3        # Run 3 rounds of alternation

set -euo pipefail

GENS_PER_ROUND=20
FIRST_VANILLA_GENS=30
MAX_ROUNDS="${1:-0}"  # 0 = unlimited

# Shared settings (matching tiered 800eval run)
COMMON_FLAGS=(
    --num-blocks 6 --hidden-dim 128
    --simulations-per-move 200 --eval-simulations 200
    --games-per-generation 200 --skip-self-play
    --enable-koth --eval-max-games 800
    --eval-explore-base 0.80
    --inference-batch-size 128 --game-threads 28
    --buffer-capacity 100000
)

TIERED_DIR="runs/long_run/scaleup_2m_tiered_800eval"
VANILLA_DIR="runs/long_run/scaleup_2m_vanilla_800eval"

run_tiered() {
    local gens=$1
    echo "=== Starting TIERED run for $gens generations ==="
    python3 python/orchestrate.py "${COMMON_FLAGS[@]}" \
        --max-generations "$gens" \
        --weights-dir "$TIERED_DIR/weights" \
        --data-dir "$TIERED_DIR/data" \
        --buffer-dir "$TIERED_DIR/buffer" \
        --log-file "$TIERED_DIR/data/training_log.jsonl"
    echo "=== TIERED run ($gens gens) complete ==="
}

run_vanilla() {
    local gens=$1
    echo "=== Starting VANILLA run for $gens generations ==="
    python3 python/orchestrate.py "${COMMON_FLAGS[@]}" \
        --disable-tier1 --disable-material \
        --max-generations "$gens" \
        --weights-dir "$VANILLA_DIR/weights" \
        --data-dir "$VANILLA_DIR/data" \
        --buffer-dir "$VANILLA_DIR/buffer" \
        --log-file "$VANILLA_DIR/data/training_log.jsonl"
    echo "=== VANILLA run ($gens gens) complete ==="
}

# Create directories
mkdir -p "$TIERED_DIR"/{weights,data,buffer}
mkdir -p "$VANILLA_DIR"/{weights,data,buffer}

# Round 1: vanilla gets extra gens (first run)
run_vanilla "$FIRST_VANILLA_GENS"
run_tiered "$GENS_PER_ROUND"

round=1
while true; do
    round=$((round + 1))
    if [ "$MAX_ROUNDS" -gt 0 ] && [ "$round" -gt "$MAX_ROUNDS" ]; then
        echo "=== Reached max rounds ($MAX_ROUNDS), stopping ==="
        break
    fi

    run_vanilla "$GENS_PER_ROUND"
    run_tiered "$GENS_PER_ROUND"
done

echo "=== All alternating runs complete ==="
