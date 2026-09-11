"""種目ごとの「周りの課題との距離」。768 次元（L2 正規化済み）で各課題の最近傍 k 件までのコサイン距離を全件計算し、
挑戦的研究（開拓・萌芽）と基盤研究(B)(C)・若手研究を比べる。分野構成の違いを除くため大区分別も出す。

出力: data/processed/knn_dist.parquet（award_number, d1, dk_mean, frac_same_cat）、reports/figures/knn_dist_by_category.png
    d1 = 最近傍までの距離, dk_mean = 最近傍 k 件の平均距離（自分自身は除く）,
    frac_chal = 近傍 k 件のうち挑戦的研究の割合（文体の同類性の確認用）

使い方: uv run python scripts/knn_dist_by_category.py
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

K = 15
BLOCK = 1024
OUT = Path("data/processed/knn_dist.parquet")
FIG = Path("reports/figures/knn_dist_by_category.png")


def main() -> None:
    emb_df = pl.read_parquet("data/processed/embeddings.parquet")
    emb = np.ascontiguousarray(emb_df["embedding"].to_numpy().astype(np.float32))
    n = emb.shape[0]
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number", "category"])
    cat = emb_df.select("award_number").join(corpus, on="award_number", how="left")["category"].to_numpy()
    chal = np.array([c.startswith("挑戦的研究") for c in cat])

    if OUT.exists():
        res = pl.read_parquet(OUT)
        print(f"{OUT} を再利用")
    else:
        d1 = np.empty(n, np.float32)
        dk = np.empty(n, np.float32)
        fchal = np.empty(n, np.float32)
        t0 = time.time()
        for s in range(0, n, BLOCK):
            e = min(s + BLOCK, n)
            sims = emb[s:e] @ emb.T  # (b, n)
            sims[np.arange(e - s), np.arange(s, e)] = -np.inf  # 自分自身を除く
            idx = np.argpartition(-sims, K, axis=1)[:, :K]
            top = np.take_along_axis(sims, idx, axis=1)
            dist = 1.0 - top
            d1[s:e] = dist.min(axis=1)
            dk[s:e] = dist.mean(axis=1)
            fchal[s:e] = chal[idx].mean(axis=1)
            if (s // BLOCK) % 20 == 0:
                print(f"  {e:,}/{n:,}  {time.time() - t0:.0f}s")
        res = pl.DataFrame({"award_number": emb_df["award_number"], "d1": d1, "dk_mean": dk, "frac_chal": fchal})
        res.write_parquet(OUT)

    import sys
    sys.path.insert(0, "src")
    from kaken_atlas.kubun import load_dai_labels

    df = res.join(corpus, on="award_number", how="left").join(load_dai_labels(), on="award_number", how="left")
    df = df.with_columns(
        pl.when(pl.col("category").str.starts_with("挑戦的研究")).then(pl.lit("挑戦的研究"))
        .when(pl.col("category").is_in(["基盤研究(B)", "基盤研究(C)"])).then(pl.lit("基盤研究(B)(C)"))
        .when(pl.col("category") == "若手研究").then(pl.lit("若手研究"))
        .otherwise(pl.lit("その他")).alias("grp")
    )
    base = chal.mean()
    print(f"\n挑戦的研究の全体比率 {base:.3f}")
    summ = (
        df.group_by("grp").agg(
            pl.len().alias("n"),
            pl.col("d1").median().alias("d1_med"),
            pl.col("dk_mean").median().alias("dk_med"),
            pl.col("dk_mean").mean().alias("dk_mean"),
            pl.col("frac_chal").mean().alias("frac_chal"),
        ).sort("grp")
    )
    print(summ)
    print("\n大区分別（dk_mean の中央値: 挑戦的 / 基盤BC / 若手）")
    by = (
        df.filter(pl.col("grp") != "その他")
        .group_by(["dai", "grp"]).agg(pl.col("dk_mean").median().alias("m"), pl.len().alias("n"))
        .pivot(on="grp", index="dai", values="m").sort("dai")
    )
    print(by)
    # 密度で揃えた比較: 挑戦的の各課題に対し、同じ大区分の基盤BC から dk_mean の分位で比べる代わりに、
    # 全体の dk_mean 分布での分位（0=最も密, 1=最も孤立）を出す
    ranks = df["dk_mean"].rank() / df.height
    df = df.with_columns(ranks.alias("q"))
    print("\n孤立度の分位（全体の中での順位, 0.5 が平均）:")
    print(df.filter(pl.col("grp") != "その他").group_by("grp").agg(pl.col("q").mean().alias("q_mean"), (pl.col("q") > 0.9).mean().alias("top10%"), (pl.col("q") < 0.1).mean().alias("bottom10%")).sort("grp"))

    # 図: 群ごとの dk_mean 分布（累積）と大区分別の中央値
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="#fcfcfb")
    colors = {"挑戦的研究": "#b2532e", "基盤研究(B)(C)": "#2a78d6", "若手研究": "#3a9a6e"}
    ax = axes[0]
    for g, c in colors.items():
        v = np.sort(df.filter(pl.col("grp") == g)["dk_mean"].to_numpy())
        ax.plot(v, np.linspace(0, 1, len(v)), color=c, lw=1.8, label=f"{g}（{len(v):,}件, 中央値 {np.median(v):.3f}）")
    ax.set_xlabel("最近傍 15 件までのコサイン距離の平均（768 次元）")
    ax.set_ylabel("累積割合")
    ax.set_xlim(np.quantile(df["dk_mean"].to_numpy(), [0.002, 0.998]))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(a) 周りの課題との距離の分布", loc="left", fontsize=12)
    ax = axes[1]
    dais = [d for d in by["dai"].to_list() if d is not None and d not in ("区分なし", "複数")]
    xpos = np.arange(len(dais))
    for j, (g, c) in enumerate(colors.items()):
        if g not in by.columns:
            continue
        vals = [(by.filter(pl.col("dai") == d)[g][0] or np.nan) for d in dais]
        ax.bar(xpos + (j - 1) * 0.27, vals, width=0.26, color=c, label=g)
    ax.set_xticks(xpos)
    ax.set_xticklabels(dais, fontsize=9)
    ax.set_ylabel("最近傍 15 件までの距離の中央値")
    ax.set_ylim(bottom=float(np.nanmin(by.filter(pl.col("dai").is_in(dais)).select(list(colors.keys())).to_numpy().astype(float))) * 0.9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("(b) 大区分別（分野構成の違いを除いた比較）", loc="left", fontsize=12)
    for a in axes:
        for s in a.spines.values():
            s.set_visible(False)
        a.set_facecolor("#fcfcfb")
    fig.suptitle("挑戦的研究は「周りの課題との距離」が長いか", x=0.02, y=0.99, ha="left", fontsize=14)
    fig.text(0.02, 0.005, "データ: KAKEN 科研費データベース（国立情報学研究所）| 2019–2025年度開始の採択課題 206,078件 | 埋め込み: cl-nagoya/ruri-v3-310m（768次元, L2正規化）| "
             "距離: 1−コサイン類似度, 最近傍は自分自身を除く全課題から | 作成: KAKEN-ATLAS (26K15524)", fontsize=7, color="#898781")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIG, dpi=170, facecolor="#fcfcfb")
    print(f"出力: {FIG}")


if __name__ == "__main__":
    main()
