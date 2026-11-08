#!/usr/bin/env bash
# Run inside the Spark container, from the repository root.
set -euo pipefail
mkdir -p runs/spark_benchmark
python scripts/prepare_spark_benchmark.py
python -m fsop.cli model fetch
python -m fsop.cli model prepare
python -c 'from fsop.data.feature_cache import build_cache; build_cache("data/sim/spark_test", batch=64, prefetch=2)'
python scripts/benchmark_training.py --cfg configs/train/spark_stage1.yaml --fit --label spark_smoke_stage1 \
  -o sim_root=data/sim/spark_test -o steps=500 -o warmup=50 -o val_every=250 -o val_batches=4 \
  -o viz_samples=0 -o log_every=50
python scripts/benchmark_training.py --cfg configs/train/spark_stage2.yaml --fit --label spark_smoke_stage2 \
  -o sim_root=data/sim/spark_test -o init_ckpt=runs/spark_smoke_stage1/ckpt/best.pt \
  -o steps=500 -o warmup=50 -o val_every=250 -o val_batches=4 -o viz_samples=0 -o log_every=50
python scripts/evaluate_spark_smoke.py
