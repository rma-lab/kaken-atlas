"""UMAP 2D 座標を大区分（A〜K）で塗った地図を描く。

出力（reports/figures/）:
1. map_dai_facets.png — 小倍数図（12パネル: A〜K + 複数所属）。各パネルで対象を
   1色ハイライト、他はグレー。色覚安全で厳密な本命図。
2. map_dai_combined.png — 11色の一覧図。全体の縮図（gestalt）用。11カテゴリ同時彩色は
   色覚識別の保証外のため、厳密な読み取りは facets 版を正とする。

割り当てルール: 課題の小区分（複数可）から大区分集合を引き、一意→その大区分、
複数にまたがる→「複数」、小区分なし→「区分なし」（両図ともグレー背景扱い）。

使い方:
    uv run python scripts/plot_map_dai.py data/processed/umap2d_nn15_md0.1.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
HILITE = "#2a78d6"  # facets 用ハイライト（パレット slot1）
OTHER = "#d8d7d0"

# 一覧図用 11色 — 意味的隣接を色相の隣接に写した「意味順色相環」。
# 大区分重心（768次元）の距離行列から最短巡回路 A→J→C→B→D→E→K→F→G→H→I→A を
# Held-Karp で求め、OKLCH (L=0.60, C=0.145) の色相環に等間隔配置（回転は
# I=赤・G=緑・A=青 の慣習アンカーへの最小二乗で決定）。近い分野ほど近い色相に
# なるため、色の混同しやすさが意味的な近さと一致する。11色同時の厳密な識別は
# 保証外なので、精読は facets 版を正とする。
DAI_COLORS = {  # OKLCH L=0.60, C=0.17（一部 gamut clip あり）
    "A": "#bf50a0", "B": "#0088db", "C": "#6473e4", "D": "#0098b6",
    "E": "#009f7d", "F": "#868700", "G": "#b66f00", "H": "#ce5601",
    "I": "#d14a63", "J": "#9b5fce", "K": "#2e9932",
}
# 大区分の通称（公式名はA〜Kのみ。所属中区分の構成から付けた便宜的な説明）
DAI_GLOSS = {
    "A": "人文学・社会科学", "B": "数物系科学", "C": "化学", "D": "工学",
    "E": "材料・応用工学", "F": "農学", "G": "生物学", "H": "薬学・基礎医学",
    "I": "医歯薬学（臨床）", "J": "情報学", "K": "環境学",
}


def load(coords_path: Path) -> pl.DataFrame:
    tab = pl.read_csv("data/reference/kubun_table.csv", schema_overrides={"sho_code": pl.Utf8})
    sho2dai: dict[str, set[str]] = {}
    for r in tab.iter_rows(named=True):
        sho2dai.setdefault(r["sho_code"], set()).add(r["dai_code"])

    coords = pl.read_parquet(coords_path)
    corpus = pl.read_parquet(
        "data/processed/corpus.parquet", columns=["award_number", "shokubun_codes"]
    )
    df = coords.join(corpus, on="award_number", how="left")

    def classify(codes) -> str:
        dais: set[str] = set()
        for c in codes:
            dais |= sho2dai.get(c, set())
        if not dais:
            return "区分なし"
        return next(iter(dais)) if len(dais) == 1 else "複数"

    return df.with_columns(
        dai=pl.Series([classify(c.to_list()) for c in df["shokubun_codes"]])
    )


def plot_facets(df: pl.DataFrame, out: Path) -> None:
    panels = [*"ABCDEFGHIJK", "複数"]
    fig, axes = plt.subplots(3, 4, figsize=(19, 14.5), facecolor=SURFACE)
    for ax, dai in zip(axes.flat, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.scatter(df["c0"], df["c1"], s=0.25, c=OTHER, alpha=0.2, linewidths=0, rasterized=True)
        sub = df.filter(pl.col("dai") == dai)
        ax.scatter(
            sub["c0"], sub["c1"], s=0.35, c=HILITE, alpha=0.45, linewidths=0, rasterized=True
        )
        gloss = DAI_GLOSS.get(dai, "複数の大区分に所属")
        ax.set_title(f"大区分{dai}〈{gloss}〉 {sub.height:,}件" if dai != "複数"
                     else f"複数所属 {sub.height:,}件", color=INK, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(
        "大区分別の分布（各パネルで当該区分をハイライト。小区分なしの課題はグレーのまま）",
        color=INK, fontsize=14, y=0.995,
    )
    fig.text(
        0.01, 0.005,
        "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | 2019–2025年度開始の採択課題 206,078件"
        "（採択時概要あり・不採択除く）| 埋め込み: cl-nagoya/ruri-v3-310m | UMAP (cosine, n_neighbors=15, seed=42)"
        " | 大区分の説明は便宜的な通称 | 作成: KAKEN-ATLAS (26K15524)",
        color=MUTED, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def plot_combined(df: pl.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 11), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rest = df.filter(~pl.col("dai").is_in(list(DAI_COLORS)))
    ax.scatter(rest["c0"], rest["c1"], s=0.3, c=OTHER, alpha=0.25, linewidths=0, rasterized=True)
    for dai, color in DAI_COLORS.items():
        sub = df.filter(pl.col("dai") == dai)
        ax.scatter(sub["c0"], sub["c1"], s=0.35, c=color, alpha=0.5, linewidths=0,
                   rasterized=True, label=f"{dai}〈{DAI_GLOSS[dai]}〉 {sub.height:,}")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False,
                    markerscale=20, labelcolor=INK, title="大区分（件数）", title_fontsize=10)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title("学術地図 × 公式大区分（一覧版。厳密な読み取りは区分別パネル図を参照）",
                 color=INK, fontsize=13, pad=12)
    ax.annotate(
        "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | 2019–2025年度開始の採択課題 206,078件"
        "（採択時概要あり・不採択除く）\n埋め込み: cl-nagoya/ruri-v3-310m（768次元）| UMAP (cosine,"
        " n_neighbors=15, seed=42) | 色相=意味的隣接順 | 大区分の説明は便宜的な通称 | 作成: KAKEN-ATLAS (26K15524)",
        xy=(0, -0.03), xycoords="axes fraction", color=MUTED, fontsize=7.5,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def main() -> None:
    df = load(Path(sys.argv[1]))
    outdir = Path("reports/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_facets(df, outdir / "map_dai_facets.png")
    plot_combined(df, outdir / "map_dai_combined.png")


if __name__ == "__main__":
    main()
