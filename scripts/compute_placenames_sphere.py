"""球面地図のキーワード地名（峰と山域）。2D 版 compute_placenames.py と同じ考え方を球面の上でそのまま行う。

緯度経度の格子は極で歪むので、球面上にほぼ均等な格子点（フィボナッチ格子）を置き、その上で
  1) 各課題を最寄りの格子点に数える（ヒストグラム）
  2) 格子点どうしのガウス重み（大円距離、幅 σ0）で平滑化する。粗い階層は同じ平滑化を n 回重ねる（幅は σ0×√n）
  3) 峰＝大円距離 σ 以内で最大、かつ基準密度（上位 1% 点）の PEAK_REL 倍以上の格子点
  4) 山域＝峰を種にした分水嶺（密度の高い順に隣接格子点へ広げる。反転地形に水を張るのと同じ）
  5) 山域ごとの特徴語は 2D 版と同じ採点（集中度 × 広さ、中心重み付け、表記ゆれ除去、上位 N_WORDS 語）
  6) 大きすぎる山域（内部に峰のない裾野の広い山）は位置で区画に分け、区画ごとに名付ける
球面の配置は 2D とは別の UMAP なので、2D の地名は流用できない。公式区分は使わない。

入力:  data/processed/umapsphere_nn15_md0.0_sp0.3.parquet（単位球面上の c0,c1,c2）、data/interim/awards.parquet（keywords）
出力:  data/processed/placenames_sphere.json            … 地図サイト用（各階層の峰の xyz・件数・語）
       data/interim/placenames_basins_sphere.parquet    … 課題ごとの山域 id（階層ごと）

使い方:
    uv run python scripts/compute_placenames_sphere.py [--sigma0 0.03] [--steps 4,1] [--grid 100000]
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from compute_placenames import (  # noqa: E402
    AWARDS, GAMMA, MASK_REL, MIN_N, N_WORDS, PEAK_REL, STOP_WORDS, feature_words, normalize_word,
)

COORDS = Path("data/processed/umapsphere_nn15_md0.0_sp0.3.parquet")
OUT_JSON = Path("data/processed/placenames_sphere.json")
OUT_BASINS = Path("data/interim/placenames_basins_sphere.parquet")

GRID_N = 150_000   # 格子点数（間隔 ≈ sqrt(4π/N) = 0.009 rad ≈ 0.52°）
SIGMA0 = 0.03      # 平滑化の基本幅（rad）。細かい階層の幅
STEPS = [4, 1]     # 階層ごとの平滑化の回数（幅 = SIGMA0×√n）。粗い順
K_NEIGHBORS = 8    # 分水嶺で使う格子の隣接数
# 峰・海の閾値の基準: 最大密度ではなく、課題のある格子点の密度の上位 REF_Q 点。球面の配置（min_dist=0）には中心が他の峰の
# 約 10 倍に突出した塊が 2 つあり、最大値基準だと周辺の峰が閾値で落ち、塊の裾野が 2 万件超の 1 山域になる（2026-09-17 実測）
REF_Q = 0.99
# 大きすぎる山域の分割: 球面の配置には、内部に小さな峰を持たない裾野の広い山が 2 つある（物性・材料と代謝・循環器。各 2 万件、
# 峰から 15° 以上に広がる）。地名 1 つでは粗いので、件数が SPLIT_N を超える山域は位置（球面上の k-means、k = ceil(n / SPLIT_N)）で
# 区画に分け、区画ごとに特徴語を付けて区画の重心に置く。階層ごとの閾値（粗い順）
SPLIT_N = [10_000, 4_000]


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n)
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(1 - z * z)
    phi = i * np.pi * (3 - np.sqrt(5))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def chord(theta: float) -> float:
    return 2 * np.sin(theta / 2)


def smoothing_matrix(tree: cKDTree, sigma: float) -> sparse.csr_matrix:
    """格子点どうしのガウス重み（大円距離 3σ まで）。対角は 1。"""
    n = tree.n
    d = tree.sparse_distance_matrix(tree, max_distance=chord(3 * sigma), output_type="coo_matrix")
    off = d.row != d.col
    theta = 2 * np.arcsin(np.clip(d.data[off] / 2, 0, 1))
    w = sparse.coo_matrix((np.exp(-theta**2 / (2 * sigma**2)).astype(np.float32), (d.row[off], d.col[off])), shape=(n, n))
    return (w + sparse.identity(n, dtype=np.float32, format="coo")).tocsr()


def find_peaks(field: np.ndarray, tree: cKDTree, grid: np.ndarray, nbr: np.ndarray, sigma: float, ref: float) -> np.ndarray:
    """隣接 K 点の中で最大の格子点を候補にし、大円距離 σ 以内でも最大のものを峰とする。密度の高い順。"""
    cand = np.nonzero((field >= field[nbr].max(axis=1)) & (field >= PEAK_REL * ref))[0]
    keep = []
    for i in cand[np.argsort(-field[cand])]:
        around = tree.query_ball_point(grid[i], chord(sigma))
        if field[i] >= field[around].max():
            keep.append(i)
    return np.array(keep, dtype=np.int64)


def watershed_graph(field: np.ndarray, nbr: np.ndarray, peaks: np.ndarray, ref: float) -> np.ndarray:
    """峰を種に、密度の高い順に隣接格子点へ広げる（反転地形の分水嶺と同じ）。戻り値: 格子点ごとの山域 id（1 始まり、0 = 海）。"""
    label = np.zeros(len(field), dtype=np.int32)
    mask = field > MASK_REL * ref
    heap: list[tuple[float, int]] = []
    for k, i in enumerate(peaks):
        label[i] = k + 1
        heapq.heappush(heap, (-float(field[i]), int(i)))
    while heap:
        _, i = heapq.heappop(heap)
        for j in nbr[i]:
            if label[j] == 0 and mask[j]:
                label[j] = label[i]
                heapq.heappush(heap, (-float(field[j]), int(j)))
    return label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=COORDS)
    ap.add_argument("--sigma0", type=float, default=SIGMA0)
    ap.add_argument("--steps", type=str, default=",".join(str(s) for s in STEPS))
    ap.add_argument("--grid", type=int, default=GRID_N)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    df = pl.read_parquet(args.coords, columns=["award_number", "c0", "c1", "c2"]).join(
        pl.read_parquet(AWARDS, columns=["award_number", "keywords"]), on="award_number", how="left")
    pts = df.select("c0", "c1", "c2").to_numpy().astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)

    grid = fibonacci_sphere(args.grid)
    tree = cKDTree(grid)
    _, node = tree.query(pts)                       # 課題 → 最寄りの格子点
    counts = np.bincount(node, minlength=args.grid).astype(np.float32)
    _, nbr = tree.query(grid, k=K_NEIGHBORS + 1)
    nbr = nbr[:, 1:]
    w = smoothing_matrix(tree, args.sigma0)
    print(f"課題 {len(df):,} 件, 格子 {args.grid:,} 点（間隔 ≈ {np.degrees(np.sqrt(4 * np.pi / args.grid)):.2f}°）, 平滑化行列 {w.nnz / 1e6:.1f}M 要素")

    ex = (
        df.select("award_number", "keywords").explode("keywords").drop_nulls("keywords")
        .with_columns(pl.col("keywords").map_elements(normalize_word, return_dtype=pl.Utf8).alias("w"))
        .filter((pl.col("w") != "") & ~pl.col("w").is_in(list(STOP_WORDS)))
        .unique(["award_number", "w"])
    )
    total = ex.group_by("w").len().rename({"len": "n_all"})

    fields: dict[int, np.ndarray] = {}
    f, done = counts, 0
    for n in sorted(set(steps)):
        for _ in range(n - done):
            f = w @ f
        fields[n], done = f, n

    basins_df = df.select("award_number")
    levels = []
    for li, n in enumerate(steps):
        sigma = args.sigma0 * np.sqrt(n)
        field = fields[n]
        ref = float(np.quantile(field[counts > 0], REF_Q))
        peaks = find_peaks(field, tree, grid, nbr, sigma, ref)
        label = watershed_graph(field, nbr, peaks, ref)
        basin = label[node]
        col = f"basin_s{sigma:.3g}"
        basins_df = basins_df.with_columns(pl.Series(col, basin))
        cnt = np.bincount(basin, minlength=len(peaks) + 1)
        peak_h = np.concatenate([[1.0], field[peaks]])
        wt = (field[node] / peak_h[basin]) ** GAMMA
        lv = basins_df.select("award_number", col).with_columns(pl.Series("wt", wt.astype(np.float64)))
        words = feature_words(ex.join(lv, on="award_number"), total, col)

        # 大きすぎる山域を位置で区画に分ける（区画 id = 山域 id × 100 + 区画番号）
        split_n = SPLIT_N[min(li, len(SPLIT_N) - 1)]
        sector = np.zeros(len(basin), dtype=np.int64)
        sector_places = {}
        for b in np.nonzero(cnt > split_n)[0]:
            if b == 0:
                continue
            m = np.nonzero(basin == b)[0]
            k_sec = int(np.ceil(cnt[b] / split_n))
            km = KMeans(n_clusters=k_sec, n_init=4, random_state=42).fit(pts[m])
            sector[m] = b * 100 + km.labels_ + 1
            for c in range(k_sec):
                ctr = pts[m[km.labels_ == c]].mean(axis=0)
                sector_places[int(b * 100 + c + 1)] = (ctr / np.linalg.norm(ctr), int((km.labels_ == c).sum()))
        sec_words = {}
        if sector_places:
            sv = pl.DataFrame({"award_number": df["award_number"], "sector": sector, "wt": np.ones(len(sector))}).filter(pl.col("sector") > 0)
            sec_words = feature_words(ex.join(sv, on="award_number"), total, "sector")

        places = []
        for sid, (ctr, n_sec) in sector_places.items():
            cand = sec_words.get(sid, [])
            m = sector == sid
            spread = float(np.degrees(np.arccos(np.clip(pts[m] @ ctr, -1, 1))).std())
            places.append(dict(id=sid, n=n_sec, x=round(float(ctr[0]), 4), y=round(float(ctr[1]), 4), z=round(float(ctr[2]), 4),
                               spread_deg=round(spread, 2), sector_of=int(sid // 100),
                               words=[c["w"] for c in cand[:N_WORDS]], cand=cand))
        for k, i in enumerate(peaks):
            b = k + 1
            if cnt[b] < args.min_n or cnt[b] > split_n:
                continue
            m = basin == b
            spread = float(np.degrees(np.arccos(np.clip(pts[m] @ grid[i], -1, 1))).std())
            cand = words.get(b, [])
            places.append(dict(id=b, n=int(cnt[b]), x=round(float(grid[i, 0]), 4), y=round(float(grid[i, 1]), 4),
                               z=round(float(grid[i, 2]), 4), spread_deg=round(spread, 2), peak=round(float(field[i]), 2),
                               words=[c["w"] for c in cand[:N_WORDS]], cand=cand))
        print(f"σ={np.degrees(sigma):.2f}°（{sigma:.3f} rad, 平滑化 {n} 回）: 峰 {len(peaks)}, 地名 {len(places)}（うち分割した区画 {len(sector_places)}）, "
              f"海 {int((basin == 0).sum()):,} 件, 山域件数 中央値 {int(np.median(cnt[1:])):,} / 最大 {int(cnt[1:].max()):,}")
        levels.append(dict(sigma=round(float(sigma), 4), smooth_steps=n, min_n=args.min_n, n_peaks=int(len(peaks)), places=places))

    OUT_JSON.write_text(json.dumps(dict(
        coords=str(args.coords), grid_n=args.grid, sigma0=args.sigma0, n_awards=len(df),
        params=dict(peak_rel=PEAK_REL, mask_rel=MASK_REL, ref_q=REF_Q, split_n=SPLIT_N, gamma=GAMMA, n_words=N_WORDS, k_neighbors=K_NEIGHBORS),
        levels=levels,
    ), ensure_ascii=False, indent=1), encoding="utf-8")
    basins_df.write_parquet(OUT_BASINS)
    print(f"→ {OUT_JSON} ({OUT_JSON.stat().st_size // 1024} KB), {OUT_BASINS}")


if __name__ == "__main__":
    main()
