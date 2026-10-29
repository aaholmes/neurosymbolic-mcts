#!/bin/bash
set -e

# Build with the candle GPU backend (needs nvcc >= 12.8 for RTX 50-series GPUs).
# For cuDNN convolutions use --features cudnn, with libcudnn on LIBRARY_PATH and LD_LIBRARY_PATH.
cargo build --release --features cuda --bin round_robin

./target/release/round_robin \
  --games-per-pair 100 --simulations 200 \
  --output tournament_results.csv \
  --model "tiered_gen0:runs/long_run/caissawary10/weights/gen_0.pt:tiered" \
  --model "tiered_gen2:runs/long_run/caissawary10/weights/gen_2.pt:tiered" \
  --model "tiered_gen5:runs/long_run/caissawary10/weights/gen_5.pt:tiered" \
  --model "tiered_gen9:runs/long_run/caissawary10/weights/gen_9.pt:tiered" \
  --model "tiered_gen14:runs/long_run/caissawary10/weights/gen_14.pt:tiered" \
  --model "tiered_gen19:runs/long_run/caissawary10/weights/gen_19.pt:tiered" \
  --model "vanilla_gen0:runs/long_run/caissawary10_vanilla/weights/gen_0.pt:vanilla" \
  --model "vanilla_gen1:runs/long_run/caissawary10_vanilla/weights/gen_1.pt:vanilla" \
  --model "vanilla_gen2:runs/long_run/caissawary10_vanilla/weights/gen_2.pt:vanilla" \
  --model "vanilla_gen8:runs/long_run/caissawary10_vanilla/weights/gen_8.pt:vanilla"
