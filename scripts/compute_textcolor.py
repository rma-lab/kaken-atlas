"""案D（弾性リング）の連続色を全課題について計算し、地図サイト用の色表を書く。

`plot_map_textcolor.py` で学習済みのリング（`data/interim/elastic_ring_nodes.npz`: 768 次元のノード座標と
色相角）を読み、各課題の色を **近傍ノードの色相単位ベクトルのカーネル重み付き平均** で決める
（向き＝色相、長さ＝円環合意度 R＝彩度。リングの反対側から等距離＝分野の狭間は灰色）。
リングの学習そのもの（SOM、初期化に大区分重心を使う）はここでは行わない。

出力: data/processed/textcolor_d.parquet
    award_number, r, g, b（sRGB 0–255, uint8）, hue_deg（色相角 0–360）, agree（円環合意度 R 0–1）
    data/processed/textcolor_legend.json（地図サイトの凡例用）
    sectors: 色相環を 12 等分した各方位の代表色と、そこに集まる課題のキーワード（特徴語。頻度ではなく
             「その方位への集中度」= 方位内件数 /（全体件数 + 30）で上位を選ぶ）
    dai:     大区分ごとの代表色（所属課題の色の OKLab 平均。分散した区分ほど灰色に寄る）

使い方:
    uv run python scripts/compute_textcolor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from plot_map_textcolor import (  # noqa: E402
    ANCHOR_CHROMA,
    EMBEDDINGS,
    RING_NPZ,
    SEED,
    TARGET_R,
    gamut_clamp,
)

import os

OUT = Path(os.environ.get("TEXTCOLOR_OUT", "data/processed/textcolor_d.parquet"))
LEGEND_OUT = OUT.with_name("textcolor_legend.json")
AWARDS = Path("data/interim/awards.parquet")
N_SECTORS = 12
N_WORDS = 4
MIN_COUNT = 30  # 方位の特徴語に採る語の最低件数（方位内）
LIGHTNESS = float(os.environ.get("TEXTCOLOR_L", "0.56"))  # OKLab L（静的図は 0.60。地図サイトは半透明で淡くなるので下げる）
# 彩度の決め方（地図サイト用、2026-09-11 決定）: 合意度 R を 5–95% タイルで 0–1 に正規化し、
# アンカー彩度の FLOOR 倍〜(FLOOR+SPAN) 倍に線形に写す。下限を高く取り、半透明でも霞まないようにする。
# 静的図（95% タイル→アンカー彩度、下限なし）は TEXTCOLOR_MODE=figure、色相のみ（彩度一定）は flat
MODE = os.environ.get("TEXTCOLOR_MODE", "web")
FLOOR = float(os.environ.get("TEXTCOLOR_FLOOR", "0.85"))
SPAN = float(os.environ.get("TEXTCOLOR_SPAN", "0.45"))

def main() -> None:
    ring = np.load(RING_NPZ)
    nodes, hues = ring["nodes"], ring["hues"]
    emb_df = pl.read_parquet(EMBEDDINGS)
    emb = emb_df["embedding"].to_numpy()

    sims = emb @ nodes.T
    d2 = np.maximum(2.0 - 2.0 * sims, 0.0)
    d2 -= d2.min(axis=1, keepdims=True)
    ux, uy = np.cos(hues), np.sin(hues)

    def resultant(h: float, sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.exp(-sub / h)
        w /= w.sum(axis=1, keepdims=True)
        vx, vy = w @ ux, w @ uy
        return np.hypot(vx, vy), np.arctan2(vy, vx)

    # カーネル幅 h: R の中央値 = TARGET_R（plot_map_textcolor.py と同じ手順）
    rng = np.random.default_rng(SEED)
    sub = d2[rng.choice(len(d2), 20_000, replace=False)]
    lo, hi = 1e-4, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(np.median(resultant(mid, sub)[0])) > TARGET_R:
            lo = mid
        else:
            hi = mid
    h = (lo + hi) / 2
    r, hue = resultant(h, d2)

    if MODE == "figure":
        chroma = r * (ANCHOR_CHROMA / np.percentile(r, 95))
    elif MODE == "flat":
        chroma = np.full_like(r, ANCHOR_CHROMA)
    else:
        lo95, hi95 = np.percentile(r, [5, 95])
        t = np.clip((r - lo95) / (hi95 - lo95), 0.0, 1.0)
        chroma = ANCHOR_CHROMA * (FLOOR + SPAN * t)
    lab = np.stack([np.full_like(chroma, LIGHTNESS), chroma * np.cos(hue), chroma * np.sin(hue)], axis=1)
    rgb = np.round(gamut_clamp(lab) * 255).astype(np.uint8)

    out = emb_df.select("award_number").with_columns(
        r=pl.Series(rgb[:, 0]),
        g=pl.Series(rgb[:, 1]),
        b=pl.Series(rgb[:, 2]),
        hue_deg=pl.Series((np.degrees(hue) % 360).astype(np.float32)),
        agree=pl.Series(r.astype(np.float32)),
    )
    out.write_parquet(OUT)
    write_legend(emb_df["award_number"], hue, lab, rgb)
    print(f"{OUT}: {out.height:,} 件, mode={MODE}, L={LIGHTNESS}, h={h:.4f}, R中央値={np.median(r):.3f}, "
          f"彩度 中央値={np.median(chroma):.3f} 最小={chroma.min():.3f} 最大={chroma.max():.3f}, "
          f"BMU平均cos={sims.max(axis=1).mean():.3f}")


def write_legend(awards: pl.Series, hue: np.ndarray, lab: np.ndarray, rgb: np.ndarray) -> None:
    """色相環 12 方位の代表色と特徴語、大区分ごとの代表色を JSON に書く。"""
    sys.path.insert(0, "src")
    from kaken_atlas.kubun import load_dai_labels

    deg = np.degrees(hue) % 360
    sector = (np.floor(deg / (360 / N_SECTORS)).astype(int)) % N_SECTORS
    df = pl.DataFrame({"award_number": awards, "sector": sector})
    kws = pl.read_parquet(AWARDS, columns=["award_number", "keywords"])
    ex = (
        df.join(kws, on="award_number", how="left")
        .explode("keywords").drop_nulls("keywords")
        .with_columns(pl.col("keywords").str.strip_chars().alias("w"))
        .filter(pl.col("w") != "")
    )
    total = ex.group_by("w").len().rename({"len": "n_all"})
    per = ex.group_by(["sector", "w"]).len().rename({"len": "n_sec"}).join(total, on="w")
    per = per.filter(pl.col("n_sec") >= MIN_COUNT).with_columns(
        (pl.col("n_sec") / (pl.col("n_all") + 30)).alias("score")
    )
    sectors = []
    for k in range(N_SECTORS):
        center = (k + 0.5) * 360 / N_SECTORS
        c_lab = np.array([[LIGHTNESS, ANCHOR_CHROMA * np.cos(np.radians(center)), ANCHOR_CHROMA * np.sin(np.radians(center))]])
        c_rgb = np.round(gamut_clamp(c_lab)[0] * 255).astype(int)
        words = (
            per.filter(pl.col("sector") == k).sort("score", descending=True).head(N_WORDS)["w"].to_list()
        )
        sectors.append(dict(hue=center, rgb=[int(v) for v in c_rgb], n=int((sector == k).sum()), words=words))

    labels = df.select("award_number").join(load_dai_labels(), on="award_number", how="left")["dai"].to_numpy()
    dai = {}
    for d in sorted(set(labels.tolist()) - {None}):
        m = labels == d
        mean_lab = np.array([[LIGHTNESS, lab[m, 1].mean(), lab[m, 2].mean()]])
        d_rgb = np.round(gamut_clamp(mean_lab)[0] * 255).astype(int)
        dai[d] = dict(rgb=[int(v) for v in d_rgb], chroma=float(np.hypot(mean_lab[0, 1], mean_lab[0, 2])), n=int(m.sum()))

    LEGEND_OUT.write_text(json.dumps(dict(sectors=sectors, dai=dai), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{LEGEND_OUT}: 方位 {N_SECTORS}, 大区分 {len(dai)}")
    for sct in sectors:
        print(f"  {sct['hue']:5.1f}° n={sct['n']:6d} {' / '.join(sct['words'])}")
    for d, v in dai.items():
        print(f"  {d}: chroma={v['chroma']:.3f} rgb={v['rgb']} n={v['n']}")


if __name__ == "__main__":
    main()
