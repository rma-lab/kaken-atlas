"""キーワード地名の検証用静的図。compute_placenames.py の出力（各階層の峰・山域・語）を密度図の上に描く。

各階層（平滑化幅 σ）ごとに 1 枚: 密度の陰影 + 山域の境界（密度の谷）+ 峰の位置に地名（語を縦に積む。
文字の大きさは山域の件数に応じる）。人が見て、汎用語が混ざっていないか、既知の場所（眼科の島、生成 AI の
密集地など）に妥当な語が付くかを判断するための図。

使い方:
    uv run python scripts/plot_placenames.py            # data/processed/placenames_2d.json の全階層
    uv run python scripts/plot_placenames.py --sphere   # 球面版（placenames_sphere.json）。6 方向の正射影
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




# ---- 球面版の検証図（compute_placenames_sphere.py の出力）: 6 方向からの正射影に地名を重ねる ----
def main_sphere() -> None:
    sph = Path("data/processed/placenames_sphere.json")
    meta = json.loads(sph.read_text(encoding="utf-8"))
    pts = pl.read_parquet(meta["coords"], columns=["c0", "c1", "c2"]).to_numpy().astype(float)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    col = pl.read_parquet("data/processed/textcolor_d.parquet", columns=["r", "g", "b"]).to_numpy() / 255.0
    rng = np.random.default_rng(0)
    sub = rng.choice(len(pts), 70000, replace=False)
    views = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
    for level in meta["levels"]:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16.5), facecolor=SURFACE)
        n_max = max(p["n"] for p in level["places"])
        for ax, (name, v) in zip(axes.ravel(), views.items()):
            v = np.array(v, float)
            up = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([0, 1.0, 0])
            ex_ = np.cross(up, v); ex_ /= np.linalg.norm(ex_); ey = np.cross(v, ex_)
            vis = pts[sub] @ v > 0
            ax.add_patch(plt.Circle((0, 0), 1, color="#f1f0ea", zorder=0))
            ax.scatter(pts[sub][vis] @ ex_, pts[sub][vis] @ ey, s=1.6, c=col[sub][vis], alpha=0.6, linewidths=0, zorder=1)
            for p in sorted(level["places"], key=lambda q: -q["n"]):
                q = np.array([p["x"], p["y"], p["z"]])
                if q @ v < 0.35:   # 縁に近い峰は別の方向の図で見る
                    continue
                ax.text(q @ ex_, q @ ey, "\n".join(p["words"]), ha="center", va="center", color=INK, linespacing=1.05,
                        fontsize=5.5 + 6 * np.sqrt(p["n"] / n_max), zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03); ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"視線 {name}", color=MUTED, fontsize=11)
        fig.suptitle(f"球面のキーワード地名（σ={np.degrees(level['sigma']):.2f}°: 峰 {level['n_peaks']}、地名 {len(level['places'])}）",
                     color=INK, fontsize=16, x=0.01, ha="left")
        fig.text(0.01, 0.005,
                 "出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工 | "
                 f"対象: 2019–2025年度開始の採択課題 {meta['n_awards']:,}件 | 埋め込み: cl-nagoya/ruri-v3-310m | "
                 "球面 UMAP (cosine→haversine, n_neighbors=15, min_dist=0, spread=0.3, seed=42)\n"
                 f"密度: フィボナッチ格子 {meta['grid_n']:,} 点のヒストグラムをガウス平滑（大円距離）。峰＝σ 以内で最大、山域＝峰を種にした分水嶺。"
                 "点は 7 万件の無作為抽出、色は内容由来の連続色。縁に近い峰は別方向の図に出す | 作成: KAKEN-ATLAS (26K15524)",
                 color=MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        out = FIG_DIR / f"placenames_sphere_s{np.degrees(level['sigma']):.2f}deg.png"
        fig.savefig(out, dpi=110, facecolor=SURFACE)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    if "--sphere" in sys.argv:
        main_sphere()
    else:
        main()
