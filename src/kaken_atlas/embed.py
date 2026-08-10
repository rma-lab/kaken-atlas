"""corpus.parquet の text 列を Ruri v3 で埋め込み、embeddings.parquet を出力する。

- モデル: cl-nagoya/ruri-v3-310m（768次元）。クラスタリング用途なので
  プレフィックスなし（意味エンコードモード）でエンコードする。
- 出力: award_number + embedding（float32 × 768、L2正規化済み）の parquet。
  自己完結なので下流は corpus.parquet と award_number で結合すればよい。
- デバイスは cuda > mps > cpu の順に自動判別。同一コードでローカルの
  スモークテストと HAKUSAN（SLURM/GPU）の本番実行を賄う。

使い方:
    uv run python -m kaken_atlas.embed --limit 100   # スモークテスト
    uv run python -m kaken_atlas.embed               # 全件 → data/processed/embeddings.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from kaken_atlas.config import DATA_PROCESSED

MODEL_NAME = "cl-nagoya/ruri-v3-310m"
EMBED_DIM = 768
DEFAULT_IN = DATA_PROCESSED / "corpus.parquet"
DEFAULT_OUT = DATA_PROCESSED / "embeddings.parquet"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_corpus(
    corpus: pl.DataFrame,
    batch_size: int = 128,
    device: str | None = None,
) -> pl.DataFrame:
    """text 列をエンコードし、award_number + embedding のフレームを返す。"""
    from sentence_transformers import SentenceTransformer

    device = device or pick_device()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"モデル: {MODEL_NAME} / device={device} / {corpus.height:,} 件")

    start = time.time()
    embeddings = model.encode(
        corpus["text"].to_list(),
        batch_size=batch_size,
        normalize_embeddings=True,  # コサイン類似・クラスタリング前提でL2正規化
        show_progress_bar=True,
    )
    elapsed = time.time() - start
    print(f"エンコード完了: {elapsed:.1f}秒 ({corpus.height / elapsed:.1f} 件/秒)")

    vectors = pl.Series(embeddings.astype("float32"), dtype=pl.Array(pl.Float32, EMBED_DIM))
    return pl.DataFrame({"award_number": corpus["award_number"], "embedding": vectors})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="コーパスを Ruri v3 で埋め込む")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="先頭N件のみ（スモークテスト用）")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu（既定は自動判別）")
    args = ap.parse_args(argv)

    corpus = pl.read_parquet(args.in_path, columns=["award_number", "text"])
    if args.limit:
        corpus = corpus.head(args.limit)
        args.out = args.out.with_stem(f"{args.out.stem}_limit{args.limit}")

    result = embed_corpus(corpus, batch_size=args.batch_size, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.out)
    print(f"完了: {result.height:,} 件 × {EMBED_DIM}次元 → {args.out}")


if __name__ == "__main__":
    main()
