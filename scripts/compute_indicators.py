"""既存の新規性・独自性・架橋性・成長性の指標（引用を使わない内容ベース）を全課題について計算する（2026-10-24）。

指標の可視化研究（メモリ indicator-visualization-idea）の第一段。各指標は 768 次元の埋め込みと kNN グラフで計算し、
地図（UMAP）の座標は使わない。出力は data/processed/indicators.parquet（1 行 1 課題、高いほど「新規・架橋・成長」側）。

  d1, dk15        孤立度: 最近傍までの cosine 距離、近傍 15 件の平均（Shibayama et al. 2021 の埋め込み距離新規性の系）
  bridge_rs       架橋性（構成）: 近傍 30 件の大区分分布の Rao–Stirling 多様性 Σ p_i p_j d_ij（d=大区分重心間の cosine 距離）
  bridge_cc       架橋性（構造）: 1 − kNN グラフ（k=15）の局所クラスタ係数（隣人同士が繋がっていない＝構造的空隙に架かる）
  growth          成長性: 近傍 50 件のうち FY2023 以降の割合を全体比で割った log2（局所的な近年の増勢）
  kw_atyp         非典型結合（Uzzi et al. 2013 の代理）: キーワード対の共起 z の 10 パーセンタイルの符号反転（高いほど珍しい組合せ）
  uot_surplus     UOT 余剰質量（Ota, Ogi ほか 2026 の RUI を年度コホートに適用）: FY t の課題群を FY<t の標準 1 万件へ
                  不均衡 Sinkhorn で輸送し、残った質量の割合（標準を N_DRAWS 回取り直した平均）。FY2019 は過去がないので欠損。

段階ごとに中間物を data/interim に置き、再実行時は再利用する。

使い方:
    uv run python scripts/compute_indicators.py            # 全段階
    uv run python scripts/compute_indicators.py knn        # 段階を指定（knn / local / keywords / uot / merge）
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from kaken_atlas.kubun import load_dai_labels  # noqa: E402

EMB = Path("data/processed/embeddings.parquet")
CORPUS = Path("data/processed/corpus.parquet")
AWARDS = Path("data/interim/awards.parquet")
KNN = Path("data/interim/knn50.npz")
LOCAL = Path("data/interim/indicators_local.parquet")
KW = Path("data/interim/indicators_kw.parquet")
UOT = Path("data/interim/indicators_uot.parquet")
UOT_META = Path("data/interim/indicators_uot_meta.json")
OUT = Path("data/processed/indicators.parquet")

K = 50
BLOCK = 1024
RECENT_FY = 2023
# UOT
PCA_DIM = 50
STD_N = 10_000
N_DRAWS = 3  # 標準の取り直し回数（1 回では課題ごとの値の再現性が Spearman 0.60 しかない。2026-10-24 実測）
CHUNK = 10_000
GRID_N = 4_000
LAMBDAS = (0.05, 0.1, 0.2, 0.5)
EPSILONS = (0.01, 0.02, 0.05)  # 0.005 は費用を平均 1 に正規化した尺度では収束しない（2026-10-24 実測）
SEED = 42


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_emb() -> tuple[pl.Series, np.ndarray]:
    df = pl.read_parquet(EMB)
    return df["award_number"], np.ascontiguousarray(np.stack(df["embedding"].to_numpy()).astype(np.float32))


def load_meta(award: pl.Series) -> pl.DataFrame:
    corpus = pl.read_parquet(CORPUS, columns=["award_number", "title", "category", "start_fy"])
    awards = pl.read_parquet(AWARDS, columns=["award_number", "keywords"])
    return (
        pl.DataFrame({"award_number": award})
        .join(corpus, on="award_number", how="left")
        .join(load_dai_labels(), on="award_number", how="left")
        .join(awards, on="award_number", how="left")
    )


# ---------------------------------------------------------------- knn
def stage_knn() -> None:
    if KNN.exists():
        log(f"{KNN} を再利用")
        return
    award, emb = load_emb()
    n = emb.shape[0]
    idx_all = np.empty((n, K), np.int32)
    dist_all = np.empty((n, K), np.float32)
    t0 = time.time()
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        sims = emb[s:e] @ emb.T
        sims[np.arange(e - s), np.arange(s, e)] = -np.inf
        part = np.argpartition(-sims, K, axis=1)[:, :K]
        top = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-top, axis=1)
        idx_all[s:e] = np.take_along_axis(part, order, axis=1)
        dist_all[s:e] = 1.0 - np.take_along_axis(top, order, axis=1)
        if (s // BLOCK) % 25 == 0:
            log(f"  knn {e:,}/{n:,}  {time.time() - t0:.0f}s")
    np.savez(KNN, idx=idx_all, dist=dist_all)
    log(f"kNN 保存 {KNN}")


# ---------------------------------------------------------------- local (孤立度・架橋性・成長性)
def stage_local() -> None:
    if LOCAL.exists():
        log(f"{LOCAL} を再利用")
        return
    award, emb = load_emb()
    meta = load_meta(award)
    z = np.load(KNN)
    idx, dist = z["idx"], z["dist"]
    n = len(award)

    d1 = dist[:, 0]
    dk15 = dist[:, :15].mean(axis=1)

    # 架橋性（構成）: 近傍 30 件の大区分の Rao–Stirling
    dai = meta["dai"].fill_null("区分なし").to_numpy()
    labels = sorted(set(dai) - {"区分なし", "複数"})
    lab_id = {lab: i for i, lab in enumerate(labels)}
    code = np.array([lab_id.get(d, -1) for d in dai], np.int32)
    # 大区分の重心は全体平均を引いてから取る（Ruri の埋め込みは異方的で、生の重心同士は cosine 距離 0.03 程度しか離れない）
    mu = emb.mean(axis=0, keepdims=True)
    cent = np.stack([(emb[code == i] - mu).mean(axis=0) for i in range(len(labels))])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    dmat = 1.0 - cent @ cent.T
    nb = code[idx[:, :30]]  # (n, 30)
    valid = nb >= 0
    counts = np.zeros((n, len(labels)), np.float32)
    for i in range(len(labels)):
        counts[:, i] = ((nb == i) & valid).sum(axis=1)
    nvalid = counts.sum(axis=1)
    p = counts / np.maximum(nvalid, 1)[:, None]
    bridge_rs = np.einsum("ni,ij,nj->n", p, dmat, p).astype(np.float32)
    bridge_rs[nvalid < 10] = np.nan

    # 架橋性（構造）: 1 − 局所クラスタ係数（k=15、有向）
    k15 = idx[:, :15]
    cc = np.empty(n, np.float32)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        J = k15[s:e]                     # (b, 15)
        NJ = k15[J]                      # (b, 15, 15)  各隣人の隣人
        hit = (NJ[:, :, :, None] == J[:, None, None, :]).any(axis=2)  # (b, 15, 15): J[j] の隣人に J[l] がいるか
        cc[s:e] = hit.sum(axis=(1, 2)) / (15 * 14)
    bridge_cc = (1.0 - cc).astype(np.float32)

    # 成長性: 近傍 50 件の近年シェアの log2 比
    fy = meta["start_fy"].to_numpy()
    recent = (fy >= RECENT_FY).astype(np.float32)
    g = recent.mean()
    share = recent[idx].sum(axis=1)
    growth = np.log2(((share + 2 * g) / (K + 2)) / g).astype(np.float32)

    pl.DataFrame({
        "award_number": award, "d1": d1, "dk15": dk15,
        "bridge_rs": bridge_rs, "bridge_cc": bridge_cc, "growth": growth,
    }).write_parquet(LOCAL)
    log(f"局所指標 保存 {LOCAL}（大区分 {len(labels)} 種、近年シェア全体 {g:.3f}）")


# ---------------------------------------------------------------- keywords (非典型結合)
def norm_kw(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().lower().replace(" ", "").replace("　", "")


def stage_keywords(min_freq: int = 5) -> None:
    if KW.exists():
        log(f"{KW} を再利用")
        return
    award, _ = load_emb()
    meta = load_meta(award)
    docs = [sorted({norm_kw(k) for k in (kws or []) if k and norm_kw(k)}) for kws in meta["keywords"].to_list()]
    freq = Counter(k for d in docs for k in d)
    keep = {k for k, c in freq.items() if c >= min_freq}
    docs = [[k for k in d if k in keep] for d in docs]
    N = len(docs)
    pair_obs: Counter = Counter()
    for d in docs:
        pair_obs.update(combinations(d, 2))
    log(f"キーワード語彙 {len(freq):,} → 頻度≥{min_freq} で {len(keep):,}、対 {len(pair_obs):,}")
    z10 = np.full(N, np.nan, np.float32)
    npairs = np.zeros(N, np.int32)
    for i, d in enumerate(docs):
        if len(d) < 2:
            continue
        zs = []
        for a, b in combinations(d, 2):
            exp = freq[a] * freq[b] / N
            zs.append((pair_obs[(a, b)] - exp) / np.sqrt(exp))
        z10[i] = np.percentile(zs, 10)
        npairs[i] = len(zs)
    pl.DataFrame({"award_number": award, "kw_atyp": -z10, "kw_npairs": npairs}).write_parquet(KW)
    log(f"非典型結合 保存 {KW}（対あり {int((npairs > 0).sum()):,} 件）")


# ---------------------------------------------------------------- UOT (年度コホート)
def uot_surplus(src: np.ndarray, std: np.ndarray, lam: float, eps: float, scale: float) -> np.ndarray:
    """src の各点（等質量）を std（等質量、総質量は src と同じ）へ不均衡 Sinkhorn で輸送し、残った割合を返す。"""
    import ot

    n, m = len(src), len(std)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    M = ((1.0 - src @ std.T) / scale).astype(np.float64)
    np.maximum(M, 0, out=M)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        T = ot.unbalanced.sinkhorn_unbalanced(a, b, M, eps, lam, method="sinkhorn_stabilized",
                                              numItermax=1000, stopThr=1e-6)
        uot_surplus.converged = not any("did not converge" in str(x.message) for x in w)
    sent = np.asarray(T).sum(axis=1)
    return np.clip((a - sent) / a, 0, 1).astype(np.float32)


def stage_uot() -> None:
    if UOT.exists():
        log(f"{UOT} を再利用")
        return
    from sklearn.decomposition import PCA

    award, emb = load_emb()
    meta = load_meta(award)
    fy = meta["start_fy"].to_numpy()
    rng = np.random.default_rng(SEED)

    # 表現: 平均を引いて PCA → 再正規化（Ruri の cosine 距離は帯が狭く異方的なため）
    X = emb - emb.mean(axis=0, keepdims=True)
    X = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=SEED).fit_transform(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    samp = rng.choice(len(X), 2000, replace=False)
    scale = float((1.0 - X[samp] @ X[samp].T).mean())
    log(f"PCA {PCA_DIM} 次元、cosine 距離の平均 {scale:.3f}（費用の単位）")

    # 論文の手順: s の分布のエントロピーが最大になる (λ, ε) を小さな格子で選ぶ（FY2023 vs FY<2023 の各 4000 件）
    g_idx = rng.choice(np.where(fy == 2023)[0], GRID_N, replace=False)
    s_idx = rng.choice(np.where(fy < 2023)[0], GRID_N, replace=False)
    best, grid = None, []
    for lam in LAMBDAS:
        for eps in EPSILONS:
            t0 = time.time()
            s = uot_surplus(X[g_idx], X[s_idx], lam, eps, scale)
            conv = uot_surplus.converged
            h, _ = np.histogram(s, bins=50, range=(0, 1))
            p = h / h.sum()
            ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            grid.append({"lambda": lam, "eps": eps, "entropy": ent, "median": float(np.median(s)), "converged": conv})
            log(f"  grid λ={lam} ε={eps}: entropy {ent:.3f}, median {np.median(s):.3f}, "
                f"{'収束' if conv else '未収束'} {time.time() - t0:.0f}s")
            if conv and (best is None or ent > best[0]):
                best = (ent, lam, eps)
    _, lam, eps = best
    log(f"採用 λ={lam} ε={eps}")

    from scipy.stats import spearmanr

    surplus = np.full(len(X), np.nan, np.float32)
    baseline, reliability = {}, {}
    for t in sorted(set(fy.tolist())):
        past = np.where(fy < t)[0]
        if len(past) < 2 * STD_N:
            continue
        grp = rng.permutation(np.where(fy == t)[0])
        draws = np.empty((N_DRAWS, len(grp)), np.float32)
        base_med, base_p75 = [], []
        for d in range(N_DRAWS):
            std_idx = rng.choice(past, STD_N, replace=False)
            base_idx = rng.choice(np.setdiff1d(past, std_idx), STD_N, replace=False)
            s_base = uot_surplus(X[base_idx], X[std_idx], lam, eps, scale)
            base_med.append(float(np.median(s_base)))
            base_p75.append(float(np.percentile(s_base, 75)))
            for c in range(0, len(grp), CHUNK):
                ch = slice(c, c + CHUNK)
                draws[d, ch] = uot_surplus(X[grp[ch]], X[std_idx], lam, eps, scale)
                if not uot_surplus.converged:
                    log(f"    警告: FY{t} draw {d} chunk {c // CHUNK} 未収束")
        surplus[grp] = draws.mean(axis=0)
        baseline[int(t)] = {"median": float(np.mean(base_med)), "p75": float(np.mean(base_p75))}
        reliability[int(t)] = float(spearmanr(draws[0], draws[1]).correlation) if N_DRAWS > 1 else None
        log(f"  FY{t}: n={len(grp):,} median surplus {np.median(surplus[grp]):.3f} "
            f"(標準 vs 標準 {baseline[int(t)]['median']:.3f}, 1 回どうしの Spearman {reliability[int(t)]})")
    pl.DataFrame({"award_number": award, "uot_surplus": surplus}).write_parquet(UOT)
    UOT_META.write_text(json.dumps({"lambda": lam, "eps": eps, "pca_dim": PCA_DIM, "cost_scale": scale,
                                    "std_n": STD_N, "n_draws": N_DRAWS, "grid": grid, "baseline": baseline,
                                    "single_draw_reliability": reliability}, ensure_ascii=False, indent=1))
    log(f"UOT 保存 {UOT}")


# ---------------------------------------------------------------- merge
def stage_merge() -> None:
    award, _ = load_emb()
    meta = load_meta(award).select(["award_number", "title", "category", "start_fy", "dai"])
    df = (meta.join(pl.read_parquet(LOCAL), on="award_number", how="left")
              .join(pl.read_parquet(KW), on="award_number", how="left")
              .join(pl.read_parquet(UOT), on="award_number", how="left"))
    df.write_parquet(OUT)
    log(f"出力 {OUT}: {df.height:,} 行 × {df.width} 列")


STAGES = {"knn": stage_knn, "local": stage_local, "keywords": stage_keywords, "uot": stage_uot, "merge": stage_merge}

if __name__ == "__main__":
    names = sys.argv[1:] or list(STAGES)
    for name in names:
        STAGES[name]()
