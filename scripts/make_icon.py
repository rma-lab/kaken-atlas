"""地図サイトのアイコン（favicon / PWA / apple-touch-icon）を生成する。

デザイン（2026-09-06 決定）：明背景の「羅針盤」。大区分11色の意味順色相環（plot_map_dai.DAI_COLORS を
巡回路 A→J→C→B→D→E→K→F→G→H→I の順に並べた円錐グラデーション）を環にし、その中で針が北東を指す。
環＝学術の全スペクトル、針＝方向・現在地。地図そのものは描かない（小サイズで潰れるため）。

    uv run scripts/make_icon.py            # docs/ に書き出し
    uv run scripts/make_icon.py --out DIR  # 確認用

出力：icon-512.png / icon-192.png / icon-maskable-512.png（環を小さめに＝Android の中央80%円安全域）/
      apple-touch-icon.png（180px、不透明）/ favicon.png（64px、角丸透過）
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_map_dai import DAI_COLORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RING_ORDER = list("AJCBDEKFGHI")  # 意味順色相環の巡回路
BG = (252, 252, 251)  # サイトの SURFACE
INK = (11, 11, 11)
INK_SOUTH = (170, 170, 168)
NEEDLE_DEG = 35  # 北東寄り
SS = 2048  # 超解像で描いて縮小


def hex2rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def conic(size: int, rot_deg: float = -90.0) -> np.ndarray:
    """11色を角度方向に並べた円錐グラデーション（隣接色を線形補間）。"""
    rgb = np.array([hex2rgb(DAI_COLORS[k]) for k in RING_ORDER], dtype=float)
    yy, xx = np.mgrid[0:size, 0:size]
    ang = (np.degrees(np.arctan2(yy - size / 2, xx - size / 2)) - rot_deg) % 360
    t = ang / 360 * len(rgb)
    i0 = np.floor(t).astype(int) % len(rgb)
    i1 = (i0 + 1) % len(rgb)
    f = (t - np.floor(t))[..., None]
    return rgb[i0] * (1 - f) + rgb[i1] * f


def rot(pts, cx, cy, deg):
    a = math.radians(deg)
    return [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
             cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in pts]


def render(scale: float = 1.0) -> Image.Image:
    """scale<1 で全体を中心に縮める（maskable 用）。"""
    img = Image.new("RGB", (SS, SS), BG)
    d = ImageDraw.Draw(img)
    c = SS / 2
    r_out, r_in = SS * 0.44 * scale, SS * 0.34 * scale
    ring = Image.new("L", (SS, SS), 0)
    rd = ImageDraw.Draw(ring)
    rd.ellipse([c - r_out, c - r_out, c + r_out, c + r_out], fill=255)
    rd.ellipse([c - r_in, c - r_in, c + r_in, c + r_in], fill=0)
    img.paste(Image.fromarray(np.clip(conic(SS), 0, 255).astype(np.uint8)), (0, 0), ring)

    length, width = SS * 0.27 * scale, SS * 0.075 * scale
    l, r = (c - width, c), (c + width, c)
    d.polygon(rot([(c, c - length), l, r], c, c, NEEDLE_DEG), fill=INK)
    d.polygon(rot([(c, c + length), l, r], c, c, NEEDLE_DEG), fill=INK_SOUTH)
    for rad, col in ((SS * 0.045 * scale, BG), (SS * 0.022 * scale, INK)):
        d.ellipse([c - rad, c - rad, c + rad, c + rad], fill=col)
    return img


def rounded(img: Image.Image, size: int, radius_ratio: float = 0.2) -> Image.Image:
    im = img.resize((size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * radius_ratio), fill=255)
    im.putalpha(m)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img = render()
    img.resize((512, 512), Image.LANCZOS).save(args.out / "icon-512.png", optimize=True)
    img.resize((192, 192), Image.LANCZOS).save(args.out / "icon-192.png", optimize=True)
    img.resize((180, 180), Image.LANCZOS).save(args.out / "apple-touch-icon.png", optimize=True)
    rounded(img, 64).save(args.out / "favicon.png", optimize=True)
    render(scale=0.82).resize((512, 512), Image.LANCZOS).save(args.out / "icon-maskable-512.png", optimize=True)
    print(f"出力: {args.out}/icon-512.png, icon-192.png, icon-maskable-512.png, apple-touch-icon.png, favicon.png")


if __name__ == "__main__":
    main()
