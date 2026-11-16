"""新設される中区分 91「量子情報」に当たる課題は、いまの地図のどこに座っているか（2026-10-26）。

制度は 91 を B・C・E・F・L の 5 大区分に分属させた。問い: 量子情報の課題は内容の上で一つの山域をなすのか、
分属どおり複数の大区分に散るのか。どの既存区分から来るのか。制度の 5 大区分で何割を覆えるか。

量子情報の課題の特定: 新表の「内容の例」に基づく語彙（厳格な語のみ。量子ドット・量子多体など物性一般の語は除く）が
題名に 1 回以上、または本文（題名＋キーワード＋概要）に 2 回以上。小区分 91010/91020/91030 への振り分けは、
各小区分の内容の例を Ruri で埋め込み、最も近いもの（参考値）。

出力: reports/kubun2028/quantum_info.md、reports/figures/kubun2028_quantum.png
使い方: uv run python scripts/kubun2028_quantum.py
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.cluster import HDBSCAN  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

matplotlib.rcParams["font.family"] = "Hiragino Sans"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
OUT = Path("reports/kubun2028/quantum_info.md")
FINDINGS = Path("reports/kubun2028/quantum_findings.md")  # 手書きの所見（本スクリプトの出力とは別に保守。あれば冒頭に差し込む）
FIGP = Path("reports/figures/kubun2028_quantum.png")
SEED = 42

STRICT = (r"量子(情報|計算|コンピュ|ビット|通信|暗号|鍵配送|誤り訂正|アルゴリズム|アニーリング|機械学習|センシング|センサ|もつれ|"
          r"エンタングル|中継|ネットワーク|インターネット|優位|超越|シミュレータ|回路|ゲート|メモリ|光源|測定|制御|技術|状態トモグラフィ)"
          r"|キュービット|qubit|Qubit|NISQ|超伝導量子|イオントラップ|トポロジカル量子計算|量子古典ハイブリッド|誤り耐性")
EXAMPLES = {
    "91010 ハードウェア": "量子計算素子、量子計測、量子通信デバイス、量子ビット制御・測定技術、超伝導系、量子光学系、冷却原子・イオン系、シリコン系",
    "91020 システム": "誤り耐性量子コンピュータシステム、量子ビット制御・測定システム、量子分散処理システム、NISQシステム、量子暗号システム、量子センシングシステム、量子コンピュータアーキテクチャ、量子古典ハイブリッドシステム、量子システムソフトウェア、量子通信・ネットワークシステム",
    "91030 基礎理論・応用": "量子計算理論、数理量子情報、誤り訂正理論、量子力学基礎論、量子暗号理論、量子機械学習、量子情報理論、量子アルゴリズム、量子ソフトウェア応用",
}
NEW_DAI_OF_91 = {"B", "C", "E", "F", "L"}


def new_dai(old_dai: str, chu_name: str) -> str:
    """旧大区分と中区分名から新大区分（A〜M）を引く（2028 改正）。"""
    if old_dai == "B":
        return "C" if ("素粒子" in chu_name or "天文" in chu_name) else "B"
    if old_dai == "C":
        return "E" if "電気電子" in chu_name else "D"
    if old_dai == "D":
        if "材料工学" in chu_name:
            return "E"
        if "原子力" in chu_name:
            return "D"
        return "F"  # 化学工学・ナノマイクロ・応用物理・人間医工学
    return {"A": "A", "E": "G", "F": "H", "G": "I", "H": "J", "I": "K", "J": "L", "K": "M"}.get(old_dai, "?")


def main() -> None:
    t0 = time.time()
    emb_df = pl.read_parquet("data/processed/embeddings.parquet")
    award = emb_df["award_number"]
    emb = np.ascontiguousarray(np.stack(emb_df["embedding"].to_numpy()).astype(np.float32))
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number", "title", "text", "category", "start_fy"])
    awards = pl.read_parquet("data/interim/awards.parquet", columns=["award_number", "shokubun_codes", "keywords"])
    meta = pl.DataFrame({"award_number": award}).join(corpus, on="award_number", how="left").join(awards, on="award_number", how="left")
    table = pl.read_csv("data/reference/kubun_table.csv", schema_overrides={"sho_code": pl.Utf8}).unique(subset=["sho_code"], keep="first")
    sho_name = dict(zip(table["sho_code"], table["sho_name_ja"]))
    chu_name = dict(zip(table["sho_code"], table["chu_name_ja"]))
    chu_code = dict(zip(table["sho_code"], table["chu_code"]))
    dai_code = dict(zip(table["sho_code"], table["dai_code"]))
    basins = pl.DataFrame({"award_number": award}).join(pl.read_parquet("data/interim/placenames_basins.parquet"), on="award_number", how="left")
    xy = pl.DataFrame({"award_number": award}).join(pl.read_parquet("data/processed/umap2d_nn15_md0.1.parquet"), on="award_number", how="left").select(["c0", "c1"]).to_numpy()
    knn = np.load("data/interim/knn50.npz")["idx"]

    def place_names(level_file):
        d = json.load(open(level_file))
        return {p["id"]: (p.get("name") or "・".join(p.get("words", []))) for p in d["places"]}
    names0 = place_names("data/processed/placenames_llm_2d_L0.json")
    names1 = place_names("data/processed/placenames_llm_2d_L1.json")

    # --- 量子情報の課題
    # 耐量子計算機暗号（量子計算機に耐える古典暗号）は量子情報ではないので、その語を消してから数える
    PQC = r"耐量子計算機暗号|耐量子暗号|ポスト量子暗号|耐量子計算機|耐量子"
    title_c = meta["title"].str.replace_all(PQC, "")
    text_c = meta["text"].str.replace_all(PQC, "")
    title_hit = title_c.str.contains(STRICT).fill_null(False).to_numpy()
    text_hits = text_c.str.count_matches(STRICT).fill_null(0).to_numpy()
    core = title_hit | (text_hits >= 2)
    idx = np.where(core)[0]
    n = len(idx)
    fy = meta["start_fy"].to_numpy()
    print(f"量子情報の課題 {n:,} 件 ({time.time() - t0:.0f}s)")

    # --- 小区分 91010/20/30 への振り分け（Ruri）
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("cl-nagoya/ruri-v3-310m", device="cpu")
    ex_emb = model.encode(list(EXAMPLES.values()), normalize_embeddings=True)
    sims = emb[idx] @ ex_emb.T
    sub = np.array(list(EXAMPLES))[sims.argmax(axis=1)]
    sub_counts = Counter(sub)
    print("振り分け", sub_counts, f"({time.time() - t0:.0f}s)")

    # --- 出自（現行の小区分・中区分・新大区分）
    codes = meta["shokubun_codes"].to_list()
    first = np.array([c[0] if c else "" for c in codes])
    rows_sho, rows_chu, rows_new = Counter(), Counter(), Counter()
    for i in idx:
        c = first[i]
        if not c:
            rows_new["（区分なし・大型種目）"] += 1
            continue
        rows_sho[c] += 1
        rows_chu[(dai_code[c], chu_code[c], chu_name[c])] += 1
        rows_new[new_dai(dai_code[c], chu_name[c])] += 1
    n_coded = sum(v for k, v in rows_new.items() if k != "（区分なし・大型種目）")
    covered = sum(v for k, v in rows_new.items() if k in NEW_DAI_OF_91)

    # --- 構造: kNN 凝集度、連結成分、HDBSCAN
    core_set = set(idx.tolist())
    k = 30
    frac_core = np.array([np.mean([j in core_set for j in knn[i, :k]]) for i in idx])
    pos = {v: p for p, v in enumerate(idx)}
    k2 = 15
    src, dst = [], []
    for i in idx:
        for j in knn[i, :k2]:
            if j in core_set:
                src.append(pos[i]); dst.append(pos[j])
    A = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    ncomp, labels = connected_components(A, directed=False)
    comp_sizes = np.sort(np.bincount(labels))[::-1]
    mu = emb.mean(axis=0, keepdims=True)
    X = PCA(n_components=50, random_state=SEED).fit_transform(emb[idx] - mu)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cl = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(X)
    ncl = cl.max() + 1
    print(f"kNN 連結成分 {ncomp}（最大 {comp_sizes[0]}）、HDBSCAN クラスタ {ncl}、ノイズ {np.mean(cl == -1):.0%} ({time.time() - t0:.0f}s)")

    # --- 山域
    b0 = basins["basin_s0.3"].to_numpy()[idx]
    b1 = basins["basin_s0.15"].to_numpy()[idx]
    top_b0 = Counter(b0.tolist()).most_common(6)
    top_b1 = Counter(b1.tolist()).most_common(8)

    # --- キーワードの特徴語（クラスタごと）
    kw_list = meta["keywords"].to_list()
    kw_all = Counter()
    for kws in kw_list:
        kw_all.update(set(kws or []))
    def top_terms(sel, m=6):
        c = Counter()
        for i in sel:
            c.update(set(kw_list[int(i)] or []))
        out = []
        for w, v in c.most_common(60):
            if v >= 3 and w not in ("量子情報", "量子計算", "量子コンピュータ"):
                out.append((w, v, v / max(kw_all[w], 1)))
        out.sort(key=lambda t: -t[1] * min(1, t[2] * 5))
        return "、".join(f"{w}({v})" for w, v, _ in out[:m])

    # --- 図
    cmap_new = {"B": "#3987e5", "C": "#184f95", "E": "#e07b39", "F": "#c23b22", "L": "#2a9d8f"}
    def col_new(i):
        c = first[i]
        if not c:
            return "#898781"
        return cmap_new.get(new_dai(dai_code[c], chu_name[c]), "#b08fd8")
    from umap import UMAP
    loc = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(emb[idx])
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5), facecolor=SURFACE)
    ax = axes[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.2, c="#d9d8d3", linewidths=0, rasterized=True)
    ax.scatter(xy[idx, 0], xy[idx, 1], s=7, c=[col_new(i) for i in idx], linewidths=0, alpha=0.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"全体地図（2D UMAP、20.6 万件）上の量子情報の課題 {n:,} 件。色＝出自の新大区分", color=INK, loc="left")
    ax = axes[1]
    ax.scatter(loc[:, 0], loc[:, 1], s=10, c=[col_new(i) for i in idx], linewidths=0, alpha=0.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("量子情報の課題だけで UMAP を再計算。色＝出自の新大区分", color=INK, loc="left")
    for key, cc in cmap_new.items():
        ax.scatter([], [], c=cc, label=f"新大区分 {key}（制度の分属先）", s=30)
    ax.scatter([], [], c="#b08fd8", label="その他の大区分（分属先にない）", s=30)
    ax.scatter([], [], c="#898781", label="区分なし（大型種目など）", s=30)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, fontsize=8, frameon=False)
    ax = axes[2]
    sub_col = {"91010 ハードウェア": "#e07b39", "91020 システム": "#2a9d8f", "91030 基礎理論・応用": "#3987e5"}
    ax.scatter(loc[:, 0], loc[:, 1], s=10, c=[sub_col[s] for s in sub], linewidths=0, alpha=0.9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("同じ配置。色＝新小区分への振り分け（内容の例との Ruri 類似、参考値）", color=INK, loc="left")
    for key, cc in sub_col.items():
        ax.scatter([], [], c=cc, label=f"{key}（{sub_counts[key]:,}）", s=30)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, fontsize=8, frameon=False)
    for a in axes:
        a.set_facecolor(SURFACE)
        for s_ in a.spines.values():
            s_.set_visible(False)
    fig.suptitle("新設「中区分 91 量子情報」に当たる課題は地図のどこにいるか", color=INK, x=0.01, ha="left", fontsize=15)
    fig.text(0.01, 0.005, "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | 対象: 2019–2025年度開始の採択課題 206,078件のうち"
             "量子情報の語彙（新表の内容の例に基づく）を題名に含むか本文に 2 回以上含むもの | 埋め込み: cl-nagoya/ruri-v3-310m | 作成: KAKEN-ATLAS (26K15524)",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(FIGP, dpi=130, facecolor=SURFACE); plt.close(fig)

    # --- レポート
    L = [f"# 新設「中区分 91 量子情報」の課題はいまの地図のどこにいるか（{n:,} 件、2026-10-26）\n",
         "特定: 新表の内容の例に基づく量子情報の語彙（量子情報・量子計算・量子コンピュータ・量子ビット・量子通信・量子暗号・量子誤り訂正・量子アルゴリズム・量子アニーリング・"
         "量子機械学習・量子センシング・量子もつれ・NISQ・超伝導量子・イオントラップ など。量子ドット・量子多体・量子光学だけの課題は含めない。耐量子計算機暗号は量子計算機に耐える古典暗号なので語を除いて数えた）が題名に 1 回以上、または本文に 2 回以上。\n"]
    L.append("## 件数の推移\n\n| FY | 件数 | 全採択に対する千分率 |\n|---|---|---|")
    for t in range(2019, 2026):  # 補正後に 2017–2018 開始となる 28 件は母数が小さいので表から外す
        nt = int(((fy == t) & core).sum()); tot = int((fy == t).sum())
        L.append(f"| {t} | {nt} | {1000 * nt / tot:.1f} |")
    cat = meta["category"].to_numpy()
    L.append("\n## 種目の内訳\n\n| 種目 | 件数 | 量子情報の中での割合 | その種目の全採択に占める千分率 |\n|---|---|---|---|")
    for c_, v in Counter(cat[idx].tolist()).most_common(10):
        L.append(f"| {c_} | {v} | {v / n:.0%} | {1000 * v / max(int((cat == c_).sum()), 1):.1f} |")
    L.append(f"\n## 出自（現行の審査区分 → 新大区分）\n\n区分あり {n_coded:,} 件のうち、制度が 91 を置く 5 大区分（B・C・E・F・L）の出身は **{covered:,} 件（{covered / n_coded:.0%}）**。\n")
    L.append("| 新大区分 | 件数 | 割合 | 91 の分属先か |\n|---|---|---|---|")
    for key, v in sorted(rows_new.items(), key=lambda kv: -kv[1]):
        L.append(f"| {key} | {v} | {v / n:.0%} | {'○' if key in NEW_DAI_OF_91 else ''} |")
    L.append("\n上位の中区分（現行）:\n\n| 旧大区分 | 中区分 | 件数 | 新大区分 |\n|---|---|---|---|")
    for (d, c, nm), v in rows_chu.most_common(14):
        L.append(f"| {d} | {c} {nm} | {v} | {new_dai(d, nm)} |")
    L.append("\n上位の小区分（現行）:\n\n| 小区分 | 名称 | 件数 | 新大区分 |\n|---|---|---|---|")
    for c, v in rows_sho.most_common(14):
        L.append(f"| {c} | {sho_name[c]} | {v} | {new_dai(dai_code[c], chu_name[c])} |")
    L.append(f"\n## 新小区分への振り分け（参考値）\n\n" + "、".join(f"{k}: {v:,}" for k, v in sub_counts.most_common()))
    L.append(f"\n## 内容の上で一つの山域か\n\n- kNN 凝集度: 量子情報の課題の近傍 30 件のうち量子情報の課題である割合の中央値 {np.median(frac_core):.0%}（平均 {frac_core.mean():.0%}）。"
             f"全体での量子情報のシェアは {n / len(award):.2%} なので、近傍は圧倒的に同類。")
    L.append(f"- 量子情報の課題だけに制限した kNN グラフ（k=15）の連結成分: {ncomp} 個。最大成分 {comp_sizes[0]:,} 件（{comp_sizes[0] / n:.0%}）、2 番目 {comp_sizes[1] if len(comp_sizes) > 1 else 0} 件。孤立点（成分サイズ 1）{int((comp_sizes == 1).sum())} 件。")
    L.append(f"- HDBSCAN（PCA 50、min_cluster_size=25）: クラスタ {ncl} 個、ノイズ {np.mean(cl == -1):.0%}。\n")
    L.append("| クラスタ | 件数 | 主な出自（新大区分） | 新小区分の振り分け | 特徴キーワード |\n|---|---|---|---|---|")
    for c_ in sorted(range(ncl), key=lambda c_: -int((cl == c_).sum())):
        sel = idx[cl == c_]
        nd = Counter(new_dai(dai_code[first[i]], chu_name[first[i]]) if first[i] else "なし" for i in sel)
        ss = Counter(sub[cl == c_])
        L.append(f"| {c_} | {len(sel)} | " + "、".join(f"{k} {v}" for k, v in nd.most_common(3)) + " | " +
                 "、".join(f"{k.split()[0]} {v}" for k, v in ss.most_common()) + f" | {top_terms(sel)} |")
    L.append("\n## 2D 地図上の山域（LLM 命名）\n\n粗い階層（σ=0.3）:\n\n| 山域 | 件数 | 割合 |\n|---|---|---|")
    for b, v in top_b0:
        L.append(f"| {names0.get(int(b), b)} | {v} | {v / n:.0%} |")
    L.append("\n細かい階層（σ=0.15）:\n\n| 山域 | 件数 | 割合 |\n|---|---|---|")
    for b, v in top_b1:
        L.append(f"| {names1.get(int(b), b)} | {v} | {v / n:.0%} |")
    L.append(f"\n図: {FIGP}（左: 全体地図上の位置、中: 量子情報だけの UMAP を出自の新大区分で彩色、右: 同じ配置を新小区分の振り分けで彩色）\n")
    if FINDINGS.exists():
        L.insert(2, FINDINGS.read_text().strip() + "\n")
    OUT.write_text("\n".join(L) + "\n")
    print(f"書き出し {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
