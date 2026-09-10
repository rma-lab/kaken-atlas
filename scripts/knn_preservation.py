"""低次元配置がどれだけ高次元の近傍関係を保っているかを測る（spread 等のパラメータ比較用）。

無作為に選んだ課題（既定 4,000 件）について、768 次元（コサイン）での近傍 k 件と、配置上（球面は弦長＝角距離と単調、
平面はユークリッド）での近傍 k 件を求め、両者の重なり率（recall@k）を平均する。1.0 なら近傍が完全に保存されている。

    uv run scripts/knn_preservation.py data/processed/umapsphere_*.parquet data/processed/umap2d_nn15_md0.1.parquet
"""
import sys
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

K = 15
N_Q = 4000
EMB = "data/processed/embeddings.parquet"


def main() -> None:
    emb = pl.read_parquet(EMB)
    X = np.stack(emb["embedding"].to_numpy()).astype(np.float32)
    ids = emb["award_number"].to_numpy()
    rng = np.random.default_rng(0)
    q = rng.choice(len(X), N_Q, replace=False)
    # 高次元の近傍（内積＝コサイン。自分自身を除く）
    hi = np.empty((N_Q, K), dtype=np.int64)
    for s in range(0, N_Q, 500):
        sims = X[q[s:s + 500]] @ X.T
        sims[np.arange(sims.shape[0]), q[s:s + 500]] = -np.inf
        hi[s:s + 500] = np.argpartition(-sims, K, axis=1)[:, :K]
    hi_sets = [set(r) for r in hi]
    print(f"{'file':52s} recall@{K}")
    for path in sys.argv[1:]:
        df = pl.read_parquet(path)
        assert (df["award_number"].to_numpy() == ids).all(), "行順が embeddings と一致しない: " + path
        cols = ["c0", "c1", "c2"] if "c2" in df.columns else ["c0", "c1"]
        Y = df.select(cols).to_numpy().astype(np.float64)
        nn = NearestNeighbors(n_neighbors=K + 1).fit(Y)
        _, lo = nn.kneighbors(Y[q])
        rec = np.mean([len(hi_sets[i] & set(lo[i, 1:])) / K for i in range(N_Q)])
        print(f"{path:52s} {rec:.3f}")


if __name__ == "__main__":
    main()
