"""埋め込みごとの粗密の差を測る。

局所密度 ∝ 1 / r_k^d（r_k = k 近傍距離、d = 空間の次元）。全点の局所密度の分布について
log10(90%点 / 10%点)（桁で見た動的レンジ）と Gini 係数を出す。球面は測地距離（弧長）で評価。
"""
import sys
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

K = 15


def contrast(X: np.ndarray, d: int, sphere: bool = False, label: str = "") -> None:
    nn = NearestNeighbors(n_neighbors=K + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    rk = dist[:, K]
    if sphere:  # 弦長 → 弧長
        rk = 2 * np.arcsin(np.clip(rk / 2, 0, 1))
    dens = 1.0 / np.maximum(rk, 1e-9) ** d
    p10, p50, p90 = np.percentile(dens, [10, 50, 90])
    s = np.sort(dens)
    n = len(s)
    gini = (2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum())
    print(f"{label:28s} n={n:,}  log10(p90/p10)={np.log10(p90 / p10):.2f}  "
          f"p90/p10={p90 / p10:,.0f}  Gini={gini:.3f}")


if __name__ == "__main__":
    base = "data/processed/"
    files = sys.argv[1:] or [
        ("umap2d_nn15_md0.1.parquet", 2, False, "2D 平面"),
        ("umap3d_nn15_md0.1.parquet", 3, False, "3D"),
        ("umapsphere_nn15_md0.1.parquet", 2, True, "球面（本番, md=0.1）"),
    ]
    for f in files:
        if isinstance(f, str):
            path, d, sphere, label = f, (2 if "sphere" in f or "2d" in f else 3), "sphere" in f, f
        else:
            path, d, sphere, label = f
            path = base + path
        df = pl.read_parquet(path)
        cols = ["c0", "c1", "c2"] if "c2" in df.columns else ["c0", "c1"]
        contrast(df.select(cols).to_numpy().astype(np.float64), d, sphere, label)
