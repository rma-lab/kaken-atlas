"""球面 UMAP の「海」の割合を測る。

単位球面上に一様乱数で試験点を撒き、最寄りの課題点までの角距離が閾値（既定 2°）を超える試験点の割合＝海の面積率。
あわせて陸（閾値以内）の面積率と、課題点の局所密度コントラストを出す。spread を振った比較用。

    uv run scripts/ocean_fraction.py data/processed/umapsphere_*.parquet
"""
import sys
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

N_TEST = 40_000
THRESH_DEG = (1.0, 2.0, 4.0)


def main() -> None:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(N_TEST, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    print(f"{'file':52s} " + " ".join(f"海({t:.0f}°)" for t in THRESH_DEG))
    for path in sys.argv[1:]:
        df = pl.read_parquet(path)
        X = df.select("c0", "c1", "c2").to_numpy().astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        nn = NearestNeighbors(n_neighbors=1).fit(X)
        chord, _ = nn.kneighbors(v)
        ang = np.degrees(2 * np.arcsin(np.clip(chord[:, 0] / 2, 0, 1)))
        fr = [np.mean(ang > t) for t in THRESH_DEG]
        print(f"{path:52s} " + " ".join(f"{f*100:6.1f}%" for f in fr))


if __name__ == "__main__":
    main()
