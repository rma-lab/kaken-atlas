"""indicators.parquet の指標どうしの一致・不一致と、種目・年度による差をまとめる（2026-10-24）。

出力:
  reports/figures/indicators_corr.png        Spearman 相関と上位 5% の Jaccard 重なり
  reports/figures/indicators_by_category.png 種目別の分布（指標ごとにパーセンタイル順位に変換）
  reports/indicators/summary.md              数表と、指標ごとの上位 10 課題（題名のみ）

使い方: uv run python scripts/analyze_indicators.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

IND = ["d1", "dk15", "bridge_rs", "bridge_cc", "growth", "kw_atyp", "uot_surplus"]
LABEL = {
    "d1": "孤立度 d1", "dk15": "孤立度 dk15", "bridge_rs": "架橋性（構成）", "bridge_cc": "架橋性（構造）",
    "growth": "成長性", "kw_atyp": "非典型結合", "uot_surplus": "UOT 余剰",
}
GRP = {
    "挑戦的研究": lambda c: c.str.starts_with("挑戦的研究"),
    "基盤研究(S)(A)": lambda c: c.is_in(["基盤研究(S)", "基盤研究(A)"]),
    "基盤研究(B)": lambda c: c == "基盤研究(B)",
    "基盤研究(C)": lambda c: c == "基盤研究(C)",
    "若手研究": lambda c: c == "若手研究",
    "学術変革・新学術": lambda c: c.str.starts_with("学術変革") | c.str.starts_with("新学術"),
    "特別研究員奨励費": lambda c: c == "特別研究員奨励費",
}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
FIG = Path("reports/figures")
OUT = Path("reports/indicators/summary.md")


def footnote(fig, n: int) -> None:
    fig.text(0.01, 0.005,
             "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | "
             f"対象: 2019–2025年度開始の採択課題 {n:,}件 | 埋め込み: cl-nagoya/ruri-v3-310m（768次元）| "
             "指標は scripts/compute_indicators.py（地図座標は使わない）| 作成: KAKEN-ATLAS (26K15524)",
             color=MUTED, fontsize=8)


def main() -> None:
    df = pl.read_parquet("data/processed/indicators.parquet")
    n = df.height
    meta = json.loads(Path("data/interim/indicators_uot_meta.json").read_text())
    lines = [f"# 指標の一致・不一致（{n:,} 件、2026-10-24）\n",
             f"UOT: λ={meta['lambda']}, ε={meta['eps']}, PCA {meta['pca_dim']} 次元, 費用尺度 {meta['cost_scale']:.3f}, 標準 {meta['std_n']:,} 件\n"]

    # 記述統計
    lines.append("\n## 記述統計\n\n| 指標 | 有効 | 中央値 | 5% | 95% |\n|---|---|---|---|---|")
    for c in IND:
        v = df[c].drop_nulls().drop_nans()
        lines.append(f"| {LABEL[c]} | {v.len():,} | {v.median():.3f} | {v.quantile(0.05):.3f} | {v.quantile(0.95):.3f} |")

    # Spearman と上位 5% の Jaccard
    X = df.select(IND).to_numpy()
    m = len(IND)
    rho = np.full((m, m), np.nan)
    jac = np.full((m, m), np.nan)
    tops = []
    for i, c in enumerate(IND):
        v = X[:, i]
        ok = ~np.isnan(v)
        thr = np.nanpercentile(v, 95)
        tops.append(set(np.where(ok & (v >= thr))[0]))
    for i in range(m):
        for j in range(m):
            ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
            rho[i, j] = spearmanr(X[ok, i], X[ok, j]).correlation
            jac[i, j] = len(tops[i] & tops[j]) / len(tops[i] | tops[j])
    lines.append("\n## Spearman 相関\n\n| | " + " | ".join(LABEL[c] for c in IND) + " |\n|---|" + "---|" * m)
    for i, c in enumerate(IND):
        lines.append(f"| {LABEL[c]} | " + " | ".join(f"{rho[i, j]:+.2f}" for j in range(m)) + " |")
    lines.append("\n## 上位 5% の重なり（Jaccard）\n\n| | " + " | ".join(LABEL[c] for c in IND) + " |\n|---|" + "---|" * m)
    for i, c in enumerate(IND):
        lines.append(f"| {LABEL[c]} | " + " | ".join(f"{jac[i, j]:.2f}" for j in range(m)) + " |")
    expected = 0.05 / (2 - 0.05)
    lines.append(f"\n独立なら Jaccard ≈ {expected:.3f}。")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=SURFACE)
    for ax, mat, title, vmin, vmax, cmap in [
        (axes[0], rho, "Spearman 相関", -1, 1, "RdBu_r"),
        (axes[1], jac, "上位 5% の重なり（Jaccard、独立なら 0.026）", 0, 0.5, "Blues"),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(m), [LABEL[c] for c in IND], rotation=40, ha="right", color=INK)
        ax.set_yticks(range(m), [LABEL[c] for c in IND], color=INK)
        for i in range(m):
            for j in range(m):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(mat[i, j]) > 0.5 * max(abs(vmin), vmax) else INK)
        ax.set_title(title, color=INK, loc="left")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("内容ベースの 7 指標は互いにどれだけ同じ課題を拾うか", color=INK, x=0.01, ha="left", fontsize=14)
    footnote(fig, n)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG / "indicators_corr.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 種目別（パーセンタイル順位）
    cat = df["category"]
    ranks = {c: df[c].rank(method="average") / df[c].is_not_nan().sum() for c in IND}
    rdf = pl.DataFrame({"category": cat, **{c: ranks[c] for c in IND}})
    lines.append("\n## 種目別の中央値（全体でのパーセンタイル順位、0.5 が平均的）\n\n| 種目 | n | " +
                 " | ".join(LABEL[c] for c in IND) + " |\n|---|---|" + "---|" * m)
    fig, axes = plt.subplots(1, m, figsize=(3.2 * m, 6), facecolor=SURFACE, sharey=True)
    for gi, (gname, f) in enumerate(GRP.items()):
        sub = rdf.filter(f(pl.col("category")))
        lines.append(f"| {gname} | {sub.height:,} | " + " | ".join(f"{sub[c].drop_nans().median():.3f}" for c in IND) + " |")
        for ax, c in zip(axes, IND):
            v = sub[c].drop_nans().to_numpy()
            parts = ax.violinplot([v], positions=[gi], widths=0.8, showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor("#3987e5"); b.set_alpha(0.6)
            parts["cmedians"].set_color(INK)
    for ax, c in zip(axes, IND):
        ax.set_title(LABEL[c], color=INK, fontsize=11)
        ax.set_xticks(range(len(GRP)), list(GRP), rotation=60, ha="right", fontsize=8, color=INK)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        ax.set_facecolor(SURFACE)
    axes[0].set_ylabel("全体でのパーセンタイル順位", color=INK)
    fig.suptitle("種目別の分布（既知群による妥当性確認: 挑戦的研究や大型種目は新規・架橋側に寄るか）", color=INK, x=0.01, ha="left", fontsize=13)
    footnote(fig, n)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIG / "indicators_by_category.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 年度別 UOT と基準線
    lines.append("\n## UOT 余剰の年度別中央値と標準 vs 標準の基準線\n\n| FY | n | 中央値 | 75% | 基準線 中央値 | 基準線 75% |\n|---|---|---|---|---|---|")
    for t, b in sorted(meta["baseline"].items()):
        sub = df.filter(pl.col("start_fy") == int(t))["uot_surplus"].drop_nans()
        lines.append(f"| {t} | {sub.len():,} | {sub.median():.3f} | {sub.quantile(0.75):.3f} | {b['median']:.3f} | {b['p75']:.3f} |")
    lines.append("\n格子探索: " + "; ".join(f"λ={g['lambda']} ε={g['eps']} H={g['entropy']:.2f}{'' if g['converged'] else '（未収束）'}" for g in meta["grid"]))

    # 上位 10 課題（題名のみ。氏名・機関は出さない）
    lines.append("\n## 指標ごとの上位 10 課題（題名のみ）\n")
    for c in IND:
        top = df.filter(pl.col(c).is_not_nan()).sort(c, descending=True).head(10)
        lines.append(f"\n### {LABEL[c]}\n")
        for r in top.iter_rows(named=True):
            lines.append(f"- {r[c]:.3f} [{r['category']} FY{r['start_fy']} {r['dai']}] {r['title']}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"書き出し {OUT}, {FIG / 'indicators_corr.png'}, {FIG / 'indicators_by_category.png'}")


if __name__ == "__main__":
    main()
