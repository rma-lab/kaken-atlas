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
    uv run python -m kaken_atlas.reduce --sphere --min-dist 0.0 --spread 0.3   # 球面（本番設定）
    uv run python -m kaken_atlas.reduce --torus --min-dist 0.0 --spread 0.5    # 平坦トーラス（試作 2026-10-22）
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from kaken_atlas.config import DATA_PROCESSED

DEFAULT_IN = DATA_PROCESSED / "embeddings.parquet"


def torus_metric():
    """平坦トーラス（周期境界つき 2 次元平面）上のユークリッド距離と勾配（UMAP 公式ドキュメント
    「Embedding to non-Euclidean spaces」の torus_euclidean_grad）。

    各軸の周期は output_metric_kwds の torus_dimensions で渡す。UMAP は初期配置を各軸 [0, 10] に
    正規化するので、周期を 10 にしておくと初期配置が基本領域にちょうど 1 回収まり、巻き付きによる
    もつれが起きない（2π だと 1.6 周巻く）。
    """
    import numba

    @numba.njit(fastmath=True)
    def torus_euclidean_grad(x, y, torus_dimensions=(10.0, 10.0)):
        distance_sqr = 0.0
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            a = abs(x[i] - y[i])
            if 2 * a < torus_dimensions[i]:
                distance_sqr += a ** 2
                g[i] = x[i] - y[i]
            else:
                distance_sqr += (torus_dimensions[i] - a) ** 2
                g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
        distance = np.sqrt(distance_sqr)
        return distance, g / (1e-6 + distance)

    return torus_euclidean_grad


def torus_xyz(uv: np.ndarray, size: float, R: float = 3.0, r: float = 1.0) -> np.ndarray:
    """平坦トーラス座標 (u, v)∈[0, size)² をドーナツ面（大半径 R・小半径 r）の xyz に曲げる。

    絵のための変換であり、外周が広く内周が狭い歪みを持つ。真の形は歪みのない周期平面のほう。
    """
    a = uv[:, 0] * (2 * np.pi / size)   # 小円（断面）方向
    b = uv[:, 1] * (2 * np.pi / size)   # 大円方向
    x = (R + r * np.cos(a)) * np.cos(b)
    y = (R + r * np.cos(a)) * np.sin(b)
    z = r * np.sin(a)
    return np.stack([x, y, z], axis=1)


def reduce_embeddings(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
    output_metric="euclidean",
    spread: float = 1.0,
    output_metric_kwds: dict | None = None,
) -> np.ndarray:
    from umap import UMAP

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        metric="cosine",
        output_metric=output_metric,
        output_metric_kwds=output_metric_kwds,
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(X)


def sphere_xyz(coords: np.ndarray) -> np.ndarray:
    """haversine 出力（極角 θ, 方位角 φ）を単位球面上の xyz に変換する（UMAP 公式ドキュメントの流儀）。

    球面には端がないため、平面 UMAP で起きる「周辺部が引き伸ばされる／切れる」歪みがない。
    """
    theta, phi = coords[:, 0], coords[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="埋め込みを UMAP で低次元化する")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=None, help="既定はパラメータ入りの自動命名")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="先頭N件のみ（動作確認用）")
    ap.add_argument("--sphere", action="store_true",
                    help="球面に埋め込む（output_metric='haversine'）。出力は単位球面上の c0,c1,c2 と極角 theta・方位角 phi")
    ap.add_argument("--spread", type=float, default=1.0,
                    help="UMAP の spread（『近い』とみなす距離スケール）。球面は一周が 2π しかないので 1.0 だと"
                         "点が一様に広がる。0.3 で平面並みの粗密が出る（2026-09-05 の2万件実験）")
    ap.add_argument("--torus", action="store_true",
                    help="平坦トーラス（周期境界つき 2 次元平面）に埋め込む。出力は基本領域 [0, size)² の u,v と、"
                         "ドーナツ面に曲げた c0,c1,c2（絵のため）")
    ap.add_argument("--torus-size", type=float, default=10.0,
                    help="トーラスの各軸の周期（既定 10 ＝ UMAP の初期配置の範囲と一致させる）")
    args = ap.parse_args(argv)
    if args.sphere and args.n_components != 2:
        ap.error("--sphere は n_components=2（球面の2パラメータ）でのみ使えます")
    if args.torus and (args.sphere or args.n_components != 2):
        ap.error("--torus は --sphere と併用できず、n_components=2 でのみ使えます")

    emb = pl.read_parquet(args.in_path)
    if args.limit:
        emb = emb.head(args.limit)
    X = np.stack(emb["embedding"].to_numpy())
    print(f"入力: {X.shape[0]:,} 件 × {X.shape[1]} 次元")

    start = time.time()
    if args.torus:
        output_metric = torus_metric()
        output_metric_kwds = {"torus_dimensions": (args.torus_size, args.torus_size)}
    else:
        output_metric = "haversine" if args.sphere else "euclidean"
        output_metric_kwds = None
    coords = reduce_embeddings(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_state,
        output_metric=output_metric,
        spread=args.spread,
        output_metric_kwds=output_metric_kwds,
    )
    print(f"UMAP 完了: {time.time() - start:.0f}秒")

    out = args.out
    if out is None:
        kind = "sphere" if args.sphere else "torus" if args.torus else f"{args.n_components}d"
        tag = f"umap{kind}_nn{args.n_neighbors}_md{args.min_dist}"
        if args.spread != 1.0:
            tag += f"_sp{args.spread}"
        if args.limit:
            tag += f"_limit{args.limit}"
        out = DATA_PROCESSED / f"{tag}.parquet"
    if args.torus:
        uv = np.mod(coords, args.torus_size)
        xyz = torus_xyz(uv, args.torus_size)
        df = pl.DataFrame({"award_number": emb["award_number"]}).with_columns(
            [pl.Series(f"c{i}", xyz[:, i].astype("float32")) for i in range(3)]
            + [pl.Series("u", uv[:, 0].astype("float32")),
               pl.Series("v", uv[:, 1].astype("float32"))]
        )
    elif args.sphere:
        xyz = sphere_xyz(coords)
        df = pl.DataFrame({"award_number": emb["award_number"]}).with_columns(
            [pl.Series(f"c{i}", xyz[:, i].astype("float32")) for i in range(3)]
            + [pl.Series("theta", coords[:, 0].astype("float32")),
               pl.Series("phi", coords[:, 1].astype("float32"))]
        )
    else:
        df = pl.DataFrame({"award_number": emb["award_number"]}).with_columns(
            [pl.Series(f"c{i}", coords[:, i].astype("float32")) for i in range(args.n_components)]
        )
    df.write_parquet(out)
    print(f"出力: {out}")


if __name__ == "__main__":
    main()
