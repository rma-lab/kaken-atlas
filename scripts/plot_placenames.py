"""キーワード地名の検証用静的図。compute_placenames.py の出力（各階層の峰・山域・語）を密度図の上に描く。

各階層（平滑化幅 σ）ごとに 1 枚: 密度の陰影 + 山域の境界（密度の谷）+ 峰の位置に地名（語を縦に積む。
文字の大きさは山域の件数に応じる）。人が見て、汎用語が混ざっていないか、既知の場所（眼科の島、生成 AI の
密集地など）に妥当な語が付くかを判断するための図。

使い方:
    uv run python scripts/plot_placenames.py            # data/processed/placenames_2d.json の全階層
出力: reports/figures/placenames_s<σ>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from compute_placenames import OUT_JSON, basins_for_sigma, density, load  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
FIG_DIR = Path("reports/figures")


def main() -> None:
    meta = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    df = load(Path(meta["coords"]))
    x, y = df["c0"].to_numpy().astype(float), df["c1"].to_numpy().astype(float)
    extent, (nx, ny) = meta["extent"], meta["grid"]
    cmap = LinearSegmentedColormap.from_list("d", ["#ffffff", "#dfe8f3", "#9ec5f4", "#3987e5"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for level in meta["levels"]:
        sigma = level["sigma"]
        field = density(x, y, extent, nx, ny, sigma)
        labels, _ = basins_for_sigma(field, sigma * nx / (extent[1] - extent[0]))
        edges = find_boundaries(labels, mode="inner")

        fig, ax = plt.subplots(figsize=(17, 15.5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.imshow(field, origin="lower", extent=extent, cmap=cmap, norm=PowerNorm(0.5), aspect="equal",
                  interpolation="bilinear")
        ey, ex_ = np.nonzero(edges)
        ax.scatter(extent[0] + (ex_ + 0.5) * (extent[1] - extent[0]) / nx,
                   extent[2] + (ey + 0.5) * (extent[3] - extent[2]) / ny, s=0.15, c="#6b7785", alpha=0.6, linewidths=0)

        places = level["places"]
        n_max = max(p["n"] for p in places)
        for p in sorted(places, key=lambda q: -q["n"]):
            size = 6.5 + 7 * np.sqrt(p["n"] / n_max)
            ax.text(p["x"], p["y"], "\n".join(p["words"]), ha="center", va="center", fontsize=size, color=INK,
                    linespacing=1.05, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")], zorder=5)
            ax.plot(p["x"], p["y"], "o", ms=2.5, color="#b2532e", zorder=6)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"キーワード地名（σ={sigma}: 峰 {level['n_peaks']}、地名 {len(places)}、"
                     f"{level['min_n']} 件未満の山域は無名）", color=INK, fontsize=15, loc="left")
        prm = meta["params"]
        fig.text(0.01, 0.005,
                 "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | "
                 f"対象: 2019–2025年度開始の採択課題 {meta['n_awards']:,}件 | 埋め込み: cl-nagoya/ruri-v3-310m | "
                 "次元削減: UMAP (cosine, n_neighbors=15, seed=42)\n"
                 f"密度: 格子ヒストグラム＋ガウス平滑（σ={sigma} 座標単位）。峰＝局所最大（最大密度の {prm['peak_rel']:.0%} 以上）、"
                 "山域＝反転密度の分水嶺。灰線＝山域の境界（密度の谷）、赤点＝峰 | "
                 f"語: 山域内のキーワードを特徴度 n_in/(n_all+{prm['score_k']})×n_in^{prm['alpha']} で順位付け、"
                 f"表記ゆれを除いて上位 {prm['n_words']} 語 | 作成: KAKEN-ATLAS (26K15524)",
                 color=MUTED, fontsize=7.5)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        out = FIG_DIR / f"placenames_s{sigma:g}.png"
        fig.savefig(out, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
