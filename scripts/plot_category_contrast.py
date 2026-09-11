"""ある種目群（既定: 挑戦的研究〈開拓・萌芽〉）の分布を、それ以外の課題の分布と比べる図。

3 パネル + 説明列：
  (a) それ以外の課題のシェア密度  (b) 対象種目のシェア密度  (c) 差分 = (b) − (a)
差分は plot_kde_years.py の偏差図と同じ流儀で、**|差| > 2σ の画素だけ着色**する（σ は両群の半分割ブートストラップで
画素ごとに推定した標本ノイズを合成）。赤＝対象種目が相対的に多い、青＝少ない。
差分の赤いピーク上位 3 箇所には、その周辺（半径 R）の対象種目課題のキーワードから特徴語を付ける。

使い方:
    uv run python scripts/plot_category_contrast.py [--category 挑戦的研究] [--out reports/figures/contrast_challenging.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.ndimage import gaussian_filter, maximum_filter  # noqa: E402

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_POS = "#2a78d6", "#b2532e"
GRID, MARGIN = 640, 0.5
SIGMA = 0.4          # 平滑幅（データ座標）
PEAK_R = 0.5         # ピーク周辺の課題を集める半径
N_PEAKS = 3
UMAP2D = Path("data/processed/umap2d_nn15_md0.1.parquet")


def share_field(x, y, m, grid_spec, sigma_px):
    nx, ny, rng_ = grid_spec
    h, _, _ = np.histogram2d(x[m], y[m], bins=[nx, ny], range=rng_)
    return gaussian_filter(h.T / m.sum(), sigma=sigma_px)


def noise_std(x, y, idx, grid_spec, sigma_px, n_splits=4, seed=42):
    """半分割ブートストラップ: E[(a-b)^2]/4 がフル場の分散推定（plot_kde_years.py と同じ）。"""
    nx, ny, _ = grid_spec
    sq = np.zeros((ny, nx))
    for s in range(n_splits):
        perm = np.random.default_rng(seed + s).permutation(idx)
        half = len(perm) // 2
        parts = [np.zeros(len(x), bool) for _ in range(2)]
        parts[0][perm[:half]] = True
        parts[1][perm[half:]] = True
        a, b = (share_field(x, y, p, grid_spec, sigma_px) for p in parts)
        sq += (a - b) ** 2
    return np.sqrt(sq / n_splits / 4)


def clean(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def find_peaks(field, extent, n, min_sep_px=60):
    """局所最大（min_sep_px 以内で最大）の上位 n 個をデータ座標で返す。"""
    mx = maximum_filter(field, size=min_sep_px)
    cand = np.argwhere((field == mx) & (field > 0))
    cand = cand[np.argsort(field[cand[:, 0], cand[:, 1]])[::-1]][:n]
    ny, nx = field.shape
    x0, x1, y0, y1 = extent
    return [(x0 + (c[1] + 0.5) / nx * (x1 - x0), y0 + (c[0] + 0.5) / ny * (y1 - y0), float(field[c[0], c[1]])) for c in cand]


def peak_words(df, mask_target, cx, cy, r, n_words=4):
    """ピーク周辺の対象課題のキーワード特徴語（周辺内件数 /（全体件数 + 20））。"""
    x, y = df["c0"].to_numpy(), df["c1"].to_numpy()
    near = ((x - cx) ** 2 + (y - cy) ** 2 <= r * r)
    sub = df.filter(pl.Series(near & mask_target)).explode("keywords").drop_nulls("keywords")
    allkw = df.explode("keywords").drop_nulls("keywords").group_by("keywords").len().rename({"len": "n_all"})
    top = (
        sub.group_by("keywords").len().rename({"len": "n_near"})
        .join(allkw, on="keywords")
        .filter(pl.col("n_near") >= 5)
        .with_columns((pl.col("n_near") / (pl.col("n_all") + 20)).alias("score"))
        .sort("score", descending=True).head(n_words)
    )
    return top["keywords"].to_list(), int((near & mask_target).sum()), int((near & ~mask_target).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="挑戦的研究", help="種目名の前方一致（複数種目をまとめる）")
    ap.add_argument("--label", default="挑戦的研究（開拓・萌芽）")
    ap.add_argument("--out", default="reports/figures/contrast_challenging.png")
    args = ap.parse_args()

    coords = pl.read_parquet(UMAP2D)
    corpus = pl.read_parquet("data/processed/corpus.parquet", columns=["award_number", "category"])
    kws = pl.read_parquet("data/interim/awards.parquet", columns=["award_number", "keywords"])
    df = coords.join(corpus, on="award_number", how="left").join(kws, on="award_number", how="left")
    x, y = df["c0"].to_numpy(), df["c1"].to_numpy()
    target = df["category"].str.starts_with(args.category).to_numpy()
    other = ~target
    n_t, n_o = int(target.sum()), int(other.sum())
    print(f"対象 {n_t:,} 件 / それ以外 {n_o:,} 件")

    x0, x1, y0, y1 = x.min() - MARGIN, x.max() + MARGIN, y.min() - MARGIN, y.max() + MARGIN
    nx = GRID
    ny = int(round(GRID * (y1 - y0) / (x1 - x0)))
    grid_spec = (nx, ny, [[x0, x1], [y0, y1]])
    sigma_px = SIGMA * nx / (x1 - x0)
    extent = [x0, x1, y0, y1]

    f_t = share_field(x, y, target, grid_spec, sigma_px)
    f_o = share_field(x, y, other, grid_spec, sigma_px)
    diff = f_t - f_o
    sd = np.sqrt(noise_std(x, y, np.where(target)[0], grid_spec, sigma_px) ** 2
                 + noise_std(x, y, np.where(other)[0], grid_spec, sigma_px) ** 2)
    sig = np.abs(diff) > 2 * sd
    masked = np.where(sig, diff, 0.0)
    # 分布の違いの大きさ: 全変動距離（0=同じ, 1=完全に別）
    tv = 0.5 * np.abs(f_t - f_o).sum() / f_o.sum()
    print(f"全変動距離 {tv:.3f}, 有意画素の割合 {sig.mean():.3f}")

    peaks_pos = find_peaks(masked, extent, N_PEAKS)
    peaks_neg = find_peaks(-masked, extent, 2)
    pos_info = [(px, py) + peak_words(df, target, px, py, PEAK_R) for px, py, _ in peaks_pos]
    neg_info = [(px, py) + peak_words(df, other, px, py, PEAK_R) for px, py, _ in peaks_neg]

    fig = plt.figure(figsize=(22, 9), facecolor=SURFACE)
    fig.suptitle(f"{args.label} はどこに多いか — それ以外の課題との分布の差", x=0.02, y=0.97, ha="left", color=INK, fontsize=17)
    axes = [fig.add_axes([0.005 + i * 0.228, 0.04, 0.225, 0.85]) for i in range(3)]
    cmap_seq = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ_BLUE)
    vmax = max(np.quantile(f_o, 0.999), np.quantile(f_t, 0.999))
    for ax, f, ttl, n in ((axes[0], f_o, f"(a) それ以外の課題", n_o), (axes[1], f_t, f"(b) {args.label}", n_t)):
        ax.imshow(f, origin="lower", extent=extent, cmap=cmap_seq, vmin=0, vmax=vmax, interpolation="bilinear")
        ax.set_title(f"{ttl}　{n:,}件（シェア密度）", color=INK, fontsize=12, loc="left")
        clean(ax)
    lim = np.quantile(np.abs(masked[sig]), 0.995) if sig.any() else 1.0
    cmap_div = LinearSegmentedColormap.from_list("div", [DIV_NEG, SURFACE, DIV_POS])
    im = axes[2].imshow(masked, origin="lower", extent=extent, cmap=cmap_div,
                        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), interpolation="bilinear")
    axes[2].contour(f_o, levels=[np.quantile(f_o, 0.6)], colors=[MUTED], linewidths=0.4, extent=extent, origin="lower", alpha=0.6)
    axes[2].set_title("(c) 差分 (b) − (a)　|差| > 2σ の画素のみ着色", color=INK, fontsize=12, loc="left")
    clean(axes[2])
    for k, (px, py, words, n_near, n_other) in enumerate(pos_info):
        axes[2].annotate(str(k + 1), xy=(px, py), xytext=(px + 0.9, py + 0.9), ha="center", va="center", color=INK, fontsize=13,
                         fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec=DIV_POS, lw=1.2),
                         arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=0, shrinkB=2))
    for k, (px, py, words, n_near, n_other) in enumerate(neg_info):
        axes[2].annotate(chr(ord("a") + k), xy=(px, py), xytext=(px - 0.9, py - 0.9), ha="center", va="center", color=INK, fontsize=13,
                         fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec=DIV_NEG, lw=1.2),
                         arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=0, shrinkB=2))
    cax = fig.add_axes([0.692, 0.22, 0.008, 0.48])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("シェア密度の差（赤=多い, 青=少ない）", color=MUTED, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.outline.set_visible(False)

    lines = [f"赤のピーク（{args.label}が相対的に多い場所）と周辺 半径{PEAK_R} の特徴語", ""]
    for k, (px, py, words, n_near, n_other) in enumerate(pos_info):
        share = n_near / max(n_near + n_other, 1) * 100
        lines.append(f"  {k + 1}. {' / '.join(words)}")
        lines.append(f"      周辺 {n_near + n_other:,} 件中 {n_near} 件が{args.label}（{share:.1f}%、全体では {n_t / (n_t + n_o) * 100:.1f}%）")
    lines += ["", "青のピーク（相対的に少ない場所）", ""]
    for k, (px, py, words, n_near, n_other) in enumerate(neg_info):
        share = n_other / max(n_near + n_other, 1) * 100  # ここでは n_near=それ以外, n_other=対象
        lines.append(f"  {chr(ord('a') + k)}. {' / '.join(words)}")
        lines.append(f"      周辺 {n_near + n_other:,} 件中 {n_other} 件が{args.label}（{share:.1f}%）")
    lines += ["", f"分布の違いの大きさ（全変動距離）: {tv:.2f}（0=同じ分布, 1=重なりなし）",
              f"|差| > 2σ となる画素は地図の {sig.mean() * 100:.1f}%"]
    fig.text(0.735, 0.90, "\n".join(lines), va="top", color=INK, fontsize=10.5, linespacing=1.55)
    fig.text(0.735, 0.19,
             "データ: KAKEN 科研費データベース（国立情報学研究所）\n2019–2025年度開始の採択課題 206,078件（採択時概要あり・不採択除く）\n"
             "埋め込み: cl-nagoya/ruri-v3-310m | UMAP (cosine, n_neighbors=15, min_dist=0.1, seed=42)\n"
             f"シェア密度: 各群の1件の質量を 1/n に揃えて σ={SIGMA} で平滑化\nσ（標本ノイズ）: 各群の半分割ブートストラップ（4回）を合成\n"
             "特徴語: 周辺の対象課題のキーワードを（周辺内件数）/（全体件数+20）で順位付け\n作成: KAKEN-ATLAS (26K15524)",
             va="top", color=MUTED, fontsize=7.5, linespacing=1.6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"出力: {out}")
    for k, (px, py, words, n_near, n_other) in enumerate(pos_info):
        print(f"  +{k + 1} ({px:.2f},{py:.2f}) {words} 対象{n_near}/周辺{n_near + n_other}")
    for k, (px, py, words, n_near, n_other) in enumerate(neg_info):
        print(f"  -{chr(ord('a') + k)} ({px:.2f},{py:.2f}) {words} それ以外{n_near}/対象{n_other}")


if __name__ == "__main__":
    main()
