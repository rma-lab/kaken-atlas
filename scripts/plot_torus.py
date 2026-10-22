"""平坦トーラス UMAP（reduce --torus）の試作図を描く（2026-10-22）。

2 枚:
1. torus_flat_<tag>.png   — 周期平面を 3×3 に敷き詰めた密度図。中央が基本領域。どこにも端がないことを見る。
2. torus_donut_<tag>.png  — ドーナツ面に曲げた 3D 散布（内容由来の連続色）。絵としての形を見る。

使い方:
    uv run python scripts/plot_torus.py data/processed/umaptorus_nn15_md0.0_sp0.5.parquet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FIG_DIR = Path("reports/figures")


def footnote(fig, n: int, params: str) -> None:
    fig.text(
        0.01, 0.005,
        "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | "
        f"対象: 2019–2025年度開始の採択課題のうち {n:,}件 | 埋め込み: cl-nagoya/ruri-v3-310m（768次元）| "
        f"平坦トーラス UMAP (cosine, n_neighbors=15, {params}, seed=42) | 作成: KAKEN-ATLAS (26K15524)",
        color=MUTED, fontsize=8,
    )


def text_colors(award_numbers: pl.Series) -> np.ndarray | None:
    """内容由来の連続色（textcolor_d.parquet、corpus と同じ行順）を課題番号で引く。無ければ None。"""
    tc = Path("data/processed/textcolor_d.parquet")
    if not tc.exists():
        return None
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number"])
    col = pl.read_parquet(tc, columns=["r", "g", "b"])
    if col.height != corpus.height:
        return None
    table = corpus.with_columns([col["r"], col["g"], col["b"]])
    joined = pl.DataFrame({"award_number": award_numbers}).join(table, on="award_number", how="left")
    rgb = joined.select(["r", "g", "b"]).to_numpy().astype(float) / 255.0
    return np.nan_to_num(rgb, nan=0.6)


def plot_flat(df: pl.DataFrame, size: float, tag: str, params: str) -> Path:
    u, v = df["u"].to_numpy(), df["v"].to_numpy()
    tiles_u = np.concatenate([u + i * size for i in (-1, 0, 1) for _ in (0, 1, 2)])
    tiles_v = np.concatenate([v + j * size for _ in (0, 1, 2) for j in (-1, 0, 1)])
    fig, ax = plt.subplots(figsize=(12, 12), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_BLUE)
    h = ax.hexbin(tiles_u, tiles_v, gridsize=330, cmap=cmap, norm=LogNorm(), linewidths=0)
    cb = fig.colorbar(h, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("課題数（対数）", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    ax.add_patch(plt.Rectangle((0, 0), size, size, fill=False, ls="--", lw=1.2, ec=INK, alpha=0.6))
    ax.text(size / 2, size + 0.15, "基本領域（破線）。周囲 8 枚は同じものの繰り返し＝端がない",
            ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xlim(-size, 2 * size)
    ax.set_ylim(-size, 2 * size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"平坦トーラス UMAP：周期平面の密度（{tag}）", color=INK, fontsize=14, loc="left")
    footnote(fig, df.height, params)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = FIG_DIR / f"torus_flat_{tag}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_donut(df: pl.DataFrame, size: float, tag: str, params: str) -> Path:
    xyz = df.select(["c0", "c1", "c2"]).to_numpy().astype(float)
    # 面の法線（裏側の点を隠すため。matplotlib の 3D 散布は奥行きで隠さない）
    a = df["u"].to_numpy() * (2 * np.pi / size)
    b = df["v"].to_numpy() * (2 * np.pi / size)
    normal = np.stack([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)], axis=1)
    rgb = text_colors(df["award_number"])
    rng = np.random.default_rng(0)
    n_show = min(120000, len(xyz))
    sub = rng.choice(len(xyz), n_show, replace=False)
    fig = plt.figure(figsize=(20, 10), facecolor=SURFACE)
    for k, (elev, azim) in enumerate([(35, -60), (75, -60)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d", facecolor=SURFACE)
        e, z = np.radians(elev), np.radians(azim)
        view = np.array([np.cos(e) * np.cos(z), np.cos(e) * np.sin(z), np.sin(e)])
        front = sub[normal[sub] @ view > 0.05]
        c = rgb[front] if rgb is not None else "#3987e5"
        ax.scatter(xyz[front, 0], xyz[front, 1], xyz[front, 2], s=1.4, c=c, alpha=0.75, linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.35))
        ax.set_axis_off()
        ax.set_title(f"仰角 {elev}°", color=MUTED, fontsize=11)
    fig.suptitle(f"平坦トーラス UMAP をドーナツ面に曲げた図（{tag}、{n_show:,}件を無作為抽出、色は内容由来の連続色、手前の面のみ）",
                 color=INK, fontsize=14, x=0.01, ha="left")
    footnote(fig, df.height, params + "；ドーナツは大半径 3・小半径 1 で曲げた絵であり、外周が広く内周が狭い歪みを持つ")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = FIG_DIR / f"torus_donut_{tag}.png"
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    path = Path(sys.argv[1])
    df = pl.read_parquet(path)
    size = float(np.ceil(max(df["u"].max(), df["v"].max())))
    tag = re.sub(r"^umaptorus_|\.parquet$", "", path.name)
    m = re.search(r"md([\d.]+)(?:_sp([\d.]+))?", path.name)
    params = f"min_dist={m.group(1)}, spread={m.group(2) or '1.0'}, 周期={size:g}"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(plot_flat(df, size, tag, params))
    print(plot_donut(df, size, tag, params))


if __name__ == "__main__":
    main()
