"""トップページ（docs/index.html）の背景に敷く点群画像を 2D UMAP から生成する。

明背景（サイトの SURFACE #fcfcfb）に大区分11色（plot_map_dai.DAI_COLORS）で全点を描く。
1px 集計で画素ごとに「件数→不透明度、色は平均」とし、密度の濃淡が出るようにしている。
大区分なし（区分なし・複数）はグレー。

    uv run scripts/make_hero.py            # docs/hero.jpg（2000×1250, JPEG q88）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_map_dai import DAI_COLORS  # noqa: E402

from kaken_atlas.kubun import load_dai_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COORDS = ROOT / "data/processed/umap2d_nn15_md0.1.parquet"
W, H = 2000, 1250
BG = np.array([252, 252, 251], dtype=float)
GREY = np.array([150, 150, 146], dtype=float)


def hex2rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/hero.jpg")
    args = ap.parse_args()

    df = pl.read_parquet(COORDS).join(load_dai_labels(), on="award_number", how="left")
    xy = df.select("c0", "c1").to_numpy().astype(float)
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - (lo + hi) / 2) / (hi - lo).max()
    xy[:, 1] *= -1
    px = xy * (H * 0.92) + np.array([W / 2, H / 2])

    keys = list(DAI_COLORS)
    rgb = np.array([hex2rgb(DAI_COLORS[k]) for k in keys], dtype=float)
    col = np.array([rgb[keys.index(d)] if d in DAI_COLORS else GREY for d in df["dai"].to_numpy()])

    ix = np.clip(px[:, 0].astype(int), 0, W - 1)
    iy = np.clip(px[:, 1].astype(int), 0, H - 1)
    cnt = np.zeros((H, W))
    acc = np.zeros((H, W, 3))
    np.add.at(cnt, (iy, ix), 1)
    for c in range(3):
        np.add.at(acc[:, :, c], (iy, ix), col[:, c])
    mean = acc / np.maximum(cnt, 1)[..., None]
    alpha = np.clip(cnt / 3.0, 0, 1) ** 0.6 * 0.75
    img = BG[None, None] * (1 - alpha[..., None]) + mean * alpha[..., None]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
        args.out, quality=88, optimize=True, progressive=True
    )
    print(f"出力: {args.out} ({df.height:,}点)")


if __name__ == "__main__":
    main()
