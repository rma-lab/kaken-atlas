#!/bin/bash
# HAKUSAN で corpus.parquet 全件を Ruri v3 で埋め込むバッチジョブ。
# 投入前提: リポジトリが ~/kaken-atlas に clone 済み・uv sync 済み・
#           data/processed/corpus.parquet を転送済み・モデルを HF キャッシュに取得済み。
# 投入:     cd ~/kaken-atlas && mkdir -p logs && sbatch scripts/hakusan_embed.sh
#SBATCH -p GPU-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH -J atlas-embed
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=takayuki@jaist.ac.jp

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$HOME/.local/bin/env"   # uv を PATH に

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export HF_HUB_OFFLINE=1         # 計算ノードは外部接続を仮定しない（モデルは事前取得）

# CUDA が見えない場合は CPU で走り続けず即失敗させる
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

uv run python -m kaken_atlas.embed --batch-size 256 --device cuda
