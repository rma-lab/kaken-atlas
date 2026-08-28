"""embeddings.parquet に UMAP をかけ、低次元座標（umap parquet）を出力する。

用途は2系統（CLAUDE.md/メモリの方法論参照）:
- 可視化用: n_components=2（既定）
- クラスタリング前段用: n_components=10〜50, min_dist=0

距離は cosine（埋め込みは L2 正規化済み）。再現性のため random_state を固定する
（UMAP は seed 固定時に並列最適化が切れて遅くなるが、本プロジェクトでは再現性を優先）。

使い方:
    uv run python -m kaken_atlas.reduce --limit 20000          # サブサンプルで感触
    uv run python -m kaken_atlas.reduce                        # 全件 2D
    uv run python -m kaken_atlas.reduce --n-neighbors 50       # 縮尺を変える
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from kaken_atlas.config import DATA_PROCESSED

DEFAULT_IN = DATA_PROCESSED / "embeddings.parquet"


def reduce_embeddings(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(X)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="埋め込みを UMAP で低次元化する")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=None, help="既定はパラメータ入りの自動命名")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="先頭N件のみ（動作確認用）")
    args = ap.parse_args(argv)

    emb = pl.read_parquet(args.in_path)
    if args.limit:
        emb = emb.head(args.limit)
    X = np.stack(emb["embedding"].to_numpy())
    print(f"入力: {X.shape[0]:,} 件 × {X.shape[1]} 次元")

    start = time.time()
    coords = reduce_embeddings(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    print(f"UMAP 完了: {time.time() - start:.0f}秒")

    out = args.out
    if out is None:
        tag = f"umap{args.n_components}d_nn{args.n_neighbors}_md{args.min_dist}"
        if args.limit:
            tag += f"_limit{args.limit}"
        out = DATA_PROCESSED / f"{tag}.parquet"
    df = pl.DataFrame({"award_number": emb["award_number"]}).with_columns(
        [pl.Series(f"c{i}", coords[:, i].astype("float32")) for i in range(args.n_components)]
    )
    df.write_parquet(out)
    print(f"出力: {out}")


if __name__ == "__main__":
    main()
