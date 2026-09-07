"""SNS 共有時のプレビュー画像（OGP、1200×630）を生成する。

明背景に 2D UMAP の点群（make_hero と同じ描画）を敷き、左上にタイトル、左下に件数と出典。
右下に羅針盤アイコン。X（twitter:card=summary_large_image）と Facebook/Slack 等の og:image 共通。

    uv run scripts/make_ogp.py            # docs/ogp.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_map_dai import DAI_COLORS  # noqa: E402

from kaken_atlas.kubun import load_dai_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COORDS = ROOT / "data/processed/umap2d_nn15_md0.1.parquet"
W, H = 1200, 630
BG = np.array([252, 252, 251], dtype=float)
GREY = np.array([150, 150, 146], dtype=float)
INK = (11, 11, 11)
MUTED = (111, 110, 105)
FONT_DIR = Path("/System/Library/Fonts")


def hex2rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"ヒラギノ角ゴシック {weight}.ttc"), size)


def render_points(scale: int = 2) -> Image.Image:
    """点群を 2 倍解像度で描いて縮小（点を細かく見せる）。地図は右寄せ。"""
    w, h = W * scale, H * scale
    df = pl.read_parquet(COORDS).join(load_dai_labels(), on="award_number", how="left")
    xy = df.select("c0", "c1").to_numpy().astype(float)
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - (lo + hi) / 2) / (hi - lo).max()
    xy[:, 1] *= -1
    px = xy * (h * 1.05) + np.array([w * 0.62, h * 0.52])

    keys = list(DAI_COLORS)
    rgb = np.array([hex2rgb(DAI_COLORS[k]) for k in keys], dtype=float)
    col = np.array([rgb[keys.index(d)] if d in DAI_COLORS else GREY for d in df["dai"].to_numpy()])
    ix = np.clip(px[:, 0].astype(int), 0, w - 1)
    iy = np.clip(px[:, 1].astype(int), 0, h - 1)
    cnt = np.zeros((h, w))
    acc = np.zeros((h, w, 3))
    np.add.at(cnt, (iy, ix), 1)
    for c in range(3):
        np.add.at(acc[:, :, c], (iy, ix), col[:, c])
    mean = acc / np.maximum(cnt, 1)[..., None]
    alpha = np.clip(cnt / 2.0, 0, 1) ** 0.6 * 0.85
    img = BG[None, None] * (1 - alpha[..., None]) + mean * alpha[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs/ogp.png")
    args = ap.parse_args()

    img = render_points()
    # 左側に文字を置くため、左からの白いグラデーションで点群を薄める
    grad = np.linspace(1.0, 0.0, W)[None, :, None]
    fade = np.clip((grad - 0.45) / 0.55, 0, 1) * 0.85  # 左端 0.85 → 中央付近で 0
    arr = np.asarray(img).astype(float)
    arr = arr * (1 - fade) + BG[None, None] * fade
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    d = ImageDraw.Draw(img)
    d.text((72, 96), "KAKEN-ATLAS", font=font("W5", 22), fill=MUTED)
    d.text((70, 136), "科研費\n学術地図", font=font("W7", 92), fill=INK, spacing=6)
    d.text((72, 372), "採択課題 206,078 件を\n研究概要の意味の近さで並べた地図", font=font("W4", 28), fill=INK, spacing=10)
    d.text((72, 470), "2019–2025年度開始 ・ 2D / 3D / 球面", font=font("W4", 22), fill=MUTED)
    d.text((72, 560), "データ: KAKEN：科学研究費助成事業データベース（国立情報学研究所）を編集・加工",
           font=font("W3", 17), fill=MUTED)
    d.text((72, 586), "rma-lab.github.io/kaken-atlas", font=font("W5", 17), fill=MUTED)

    icon = Image.open(ROOT / "docs/icon-512.png").convert("RGBA").resize((96, 96), Image.LANCZOS)
    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, 95, 95], radius=20, fill=255)
    img.paste(icon, (W - 96 - 56, H - 96 - 48), mask)

    img.save(args.out, optimize=True)
    print(f"出力: {args.out}")


if __name__ == "__main__":
    main()
