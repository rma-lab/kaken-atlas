"""UMAP 2D 座標から初回地図（PNG）を描く。

2枚構成:
1. density.png  — 全課題の密度マップ（構造を見る主役。単色シーケンシャル）
2. category.png — 主要3種目（基盤C/若手/基盤B）＋その他の分布

配色は dataviz 基準パレット（検証済み）。206k点の散布は全ペア比較になるため
カテゴリ色は3系列まで、残りは無彩色の Other に畳む。

使い方:
    uv run python scripts/plot_map.py data/processed/umap2d_nn15_md0.1.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"  # macOS の日本語フォント
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# dataviz 基準パレット（light mode, validated）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = {"基盤研究(C)": "#2a78d6", "若手研究": "#eb6834", "基盤研究(B)": "#1baf7a"}
OTHER = "#c3c2b7"


def load(coords_path: Path) -> pl.DataFrame:
    coords = pl.read_parquet(coords_path)
    corpus = pl.read_parquet("data/processed/corpus.parquet").select(["award_number", "category"])
    return coords.join(corpus, on="award_number", how="left")


def plot_density(df: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_BLUE)
    h = ax.hexbin(df["c0"], df["c1"], gridsize=220, cmap=cmap, norm=LogNorm(), linewidths=0)
    cb = fig.colorbar(h, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("課題数（対数）", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    ax.set_title(
        "科研費 採択課題の意味空間地図（2019–2025年度・206,078件）",
        color=INK, fontsize=13, pad=12,
    )
    _clean_axes(ax)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def plot_category(df: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    other = df.filter(~pl.col("category").is_in(list(CAT)))
    ax.scatter(other["c0"], other["c1"], s=0.4, c=OTHER, alpha=0.25, linewidths=0, rasterized=True)
    for name, color in CAT.items():
        sub = df.filter(pl.col("category") == name)
        ax.scatter(
            sub["c0"], sub["c1"], s=0.4, c=color, alpha=0.35, linewidths=0,
            rasterized=True, label=f"{name}（{sub.height:,}件）",
        )
    leg = ax.legend(
        loc="upper right", fontsize=9, frameon=False, markerscale=18, labelcolor=INK,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title("主要種目の分布（その他はグレー）", color=INK, fontsize=13, pad=12)
    _clean_axes(ax)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def _clean_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate(
        "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | 2019–2025年度開始の採択課題 206,078件"
        "（採択時概要あり・不採択除く）\n埋め込み: cl-nagoya/ruri-v3-310m（768次元）| 次元削減: UMAP"
        " (cosine, n_neighbors=15, seed=42) | 作成: KAKEN-ATLAS (26K15524)",
        xy=(0, -0.03), xycoords="axes fraction", color=MUTED, fontsize=7.5,
    )


def main() -> None:
    coords_path = Path(sys.argv[1])
    df = load(coords_path)
    outdir = Path("reports/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_density(df, outdir / "map_density.png")
    plot_category(df, outdir / "map_category.png")


if __name__ == "__main__":
    main()
