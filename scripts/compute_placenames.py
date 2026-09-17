"""データ由来の地名（キーワード地名）を 2D 地図の「峰と山域」ごとに計算する。

考え方（2026-09-17 決定）:
  地図上の課題の密度を地形と見て、密度の山頂（局所最大）を「峰」、峰から裾野まで谷で隣と分かれる
  ひとまとまりを「山域」と呼ぶ。山域は、密度を上下反転した地形に分水嶺法（watershed）を適用して得る
  （反転するので窪み＝密度の山。境界は密度の谷＝地図の空白）。山域ごとに所属課題のキーワードから
  特徴語（頻度でなく特徴度。「色の見方」の 12 方位の特徴語と同じスコア）を選び、峰の位置に置く地名とする。
  公式区分は一切使わない（公開版は人が線を引く前の地形を主役にする方針）。

階層: 平滑化の幅 σ を変えて粗い峰（全体〜中間のズーム向け）と細かい峰（拡大時向け）を別々に計算する。
  どの σ が良いかは静的図（scripts/plot_placenames.py）を見て決める。

入力:  data/processed/umap2d_nn15_md0.1.parquet（c0, c1）、data/interim/awards.parquet（keywords）
出力:  data/processed/placenames_2d.json      … 地図サイト用（各階層の峰の座標・件数・語）
       data/interim/placenames_basins.parquet  … 課題ごとの山域 id（階層ごと。静的図と R9 のクラスタ命名用）

使い方:
    uv run python scripts/compute_placenames.py [--sigmas 0.6,0.3] [--coords <parquet>]
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

COORDS = Path("data/processed/umap2d_nn15_md0.1.parquet")
AWARDS = Path("data/interim/awards.parquet")
OUT_JSON = Path("data/processed/placenames_2d.json")
OUT_BASINS = Path("data/interim/placenames_basins.parquet")

GRID = 640      # 密度場の格子解像度（長辺ピクセル数）。plot_kde_years.py と同じ
MARGIN = 0.5    # 格子の余白（座標単位）
SIGMAS = [0.3, 0.15]         # 階層ごとの平滑化幅（座標単位）。粗い順（2026-09-17 静的図で選定: 42 峰と 129 峰）
PEAK_REL = 0.03              # 峰と認める密度の下限（最大密度に対する比）。海の上の微小な山を除く
MASK_REL = 0.005             # 山域を割り当てる密度の下限（これ未満は「海」で未割り当て）
MIN_N = 150                  # 地名を付ける山域の最低件数
N_WORDS = 2                  # 地名に使う語数（2026-09-17 ユーザ決定: 2 語）
N_CAND = 8                   # JSON に残す候補語数（後で語数を変えられるように）
SCORE_K = 30                 # 特徴度 = n_in / (n_all + K)。凡例の 12 方位と同じ
ALPHA = 0.5                  # 広さの重み: 特徴度 × n_in^ALPHA。0 なら凡例と同じ（狭い専門語に寄る）、1 なら頻度に寄る
MIN_FRAC = 0.01              # 山域内で語が出現する最低割合（かつ最低 MIN_COUNT 件）
MIN_COUNT = 5
# 中心重み付け: 各課題の重み = (その位置の密度 / 峰の密度)^GAMMA。山頂の課題は 1、裾野（隣の峰との鞍部）は小さい。
# 縁の課題は架橋的で隣の峰にもまたがるため、峰らしさを語に反映させる（ユーザ提案 2026-09-17）。GAMMA=0 で重みなし
GAMMA = 1.0
# 除外語: 分野を表さない汎用語。静的図を見て育てる
STOP_WORDS = {"その他", "研究", "開発", "解析", "分析", "評価", "調査", "検討", "モデル", "システム", "データ",
              "メカニズム", "機序", "教育", "支援", "制御", "測定", "設計", "構造", "機能", "手法", "方法", "効果"}


def normalize_word(w: str) -> str:
    w = unicodedata.normalize("NFKC", w).strip()
    if w.isascii():
        w = w.lower()
    return w


def load(coords_path: Path) -> pl.DataFrame:
    coords = pl.read_parquet(coords_path)
    kws = pl.read_parquet(AWARDS, columns=["award_number", "keywords"])
    return coords.join(kws, on="award_number", how="left")


def grid_spec(x: np.ndarray, y: np.ndarray) -> tuple[list[float], int, int]:
    x0, x1 = float(x.min() - MARGIN), float(x.max() + MARGIN)
    y0, y1 = float(y.min() - MARGIN), float(y.max() + MARGIN)
    nx = GRID
    ny = int(round(GRID * (y1 - y0) / (x1 - x0)))
    return [x0, x1, y0, y1], nx, ny


def density(x: np.ndarray, y: np.ndarray, extent: list[float], nx: int, ny: int, sigma: float) -> np.ndarray:
    """格子ヒストグラム＋ガウス平滑。戻り値は field[y, x]（件/ピクセル相当）。"""
    x0, x1, y0, y1 = extent
    h, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[[x0, x1], [y0, y1]])
    sigma_px = sigma * nx / (x1 - x0)
    return gaussian_filter(h.T, sigma=sigma_px)


def pixel_index(x: np.ndarray, y: np.ndarray, extent: list[float], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = extent
    ix = np.clip(((x - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / (y1 - y0) * ny).astype(int), 0, ny - 1)
    return ix, iy


def basins_for_sigma(field: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """峰（局所最大）を検出し、反転密度の分水嶺で山域ラベル画像を作る。戻り値: (labels[y,x], peaks[k]=(iy,ix))。
    labels は 1 始まり（0 = 海・未割り当て）。peaks[k] がラベル k+1 の峰。"""
    peaks = peak_local_max(
        field, min_distance=max(1, int(round(sigma_px))), threshold_abs=PEAK_REL * field.max(), exclude_border=False,
    )
    markers = np.zeros(field.shape, dtype=np.int32)
    for k, (iy, ix) in enumerate(peaks):
        markers[iy, ix] = k + 1
    labels = watershed(-field, markers=markers, mask=field > MASK_REL * field.max())
    return labels, peaks


def feature_words(ex: pl.DataFrame, total: pl.DataFrame, basin_col: str) -> dict[int, list[dict]]:
    """山域ごとの特徴語候補（上位 N_CAND）。ex = 課題×語（正規化済み w、山域 id 列と中心重み wt 列付き）。
    n_in は素の件数（最低件数の判定用）、w_in は中心重み付きの件数（採点用）。"""
    per = (
        ex.group_by([basin_col, "w"]).agg(pl.len().alias("n_in"), pl.col("wt").sum().alias("w_in"))
        .join(total, on="w")
    )
    n_basin = ex.group_by(basin_col).agg(pl.col("award_number").n_unique().alias("n_b"))
    per = per.join(n_basin, on=basin_col)
    per = per.filter(
        (pl.col("n_in") >= MIN_COUNT) & (pl.col("n_in") >= pl.col("n_b") * MIN_FRAC)
    ).with_columns((pl.col("w_in") / (pl.col("n_all") + SCORE_K) * pl.col("w_in") ** ALPHA).alias("score"))
    out: dict[int, list[dict]] = {}
    for (b,), g in per.sort("score", descending=True).group_by([basin_col], maintain_order=True):
        cand: list[dict] = []
        for r in g.iter_rows(named=True):
            if any(similar(r["w"], c["w"]) for c in cand):
                continue  # 同じ地名の中の表記ゆれ（嚥下・嚥下障害、胃癌・胃がん）は 1 語に
            cand.append(dict(w=r["w"], n=int(r["n_in"]), n_all=int(r["n_all"]), score=round(float(r["score"]), 3)))
            if len(cand) >= N_CAND:
                break
        out[int(b)] = cand
    return out


def similar(a: str, b: str) -> bool:
    """一方が他方を含む、または文字 2-gram の重なりが大きい（がん/癌の違いも吸収）。"""
    a2, b2 = a.replace("がん", "癌").replace("・", ""), b.replace("がん", "癌").replace("・", "")
    if a2 in b2 or b2 in a2:
        return True
    ga = {a2[i:i + 2] for i in range(len(a2) - 1)}
    gb = {b2[i:i + 2] for i in range(len(b2) - 1)}
    return bool(ga and gb) and len(ga & gb) / min(len(ga), len(gb)) >= 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=COORDS)
    ap.add_argument("--sigmas", type=str, default=",".join(str(s) for s in SIGMAS))
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]

    df = load(args.coords)
    x, y = df["c0"].to_numpy().astype(float), df["c1"].to_numpy().astype(float)
    extent, nx, ny = grid_spec(x, y)
    ix, iy = pixel_index(x, y, extent, nx, ny)
    print(f"課題 {len(df):,} 件, 格子 {nx}×{ny}, 範囲 {extent}")

    ex = (
        df.select("award_number", "keywords").explode("keywords").drop_nulls("keywords")
        .with_columns(pl.col("keywords").map_elements(normalize_word, return_dtype=pl.Utf8).alias("w"))
        .filter((pl.col("w") != "") & ~pl.col("w").is_in(list(STOP_WORDS)))
        .unique(["award_number", "w"])
    )
    total = ex.group_by("w").len().rename({"len": "n_all"})

    basins_df = df.select("award_number")
    levels = []
    for li, sigma in enumerate(sigmas):
        field = density(x, y, extent, nx, ny, sigma)
        sigma_px = sigma * nx / (extent[1] - extent[0])
        labels, peaks = basins_for_sigma(field, sigma_px)
        basin = labels[iy, ix].astype(np.int32)        # 課題ごとの山域 id（0 = 海）
        col = f"basin_s{sigma:g}"
        basins_df = basins_df.with_columns(pl.Series(col, basin))
        counts = np.bincount(basin, minlength=len(peaks) + 1)
        peak_h = np.concatenate([[1.0], field[peaks[:, 0], peaks[:, 1]]])   # 山域 id → 峰の密度（id 0 はダミー）
        wt = (field[iy, ix] / peak_h[basin]) ** GAMMA
        lv = basins_df.select("award_number", col).with_columns(pl.Series("wt", wt))
        words = feature_words(ex.join(lv, on="award_number"), total, col)

        places = []
        for k, (py, px) in enumerate(peaks):
            b = k + 1
            n = int(counts[b])
            if n < args.min_n:
                continue
            m = basin == b
            cand = words.get(b, [])
            places.append(dict(
                id=b, n=n,
                x=round(extent[0] + (px + 0.5) * (extent[1] - extent[0]) / nx, 3),
                y=round(extent[2] + (py + 0.5) * (extent[3] - extent[2]) / ny, 3),
                sx=round(float(x[m].std()), 3), sy=round(float(y[m].std()), 3),   # 山域の広がり（標準偏差）
                peak=round(float(field[py, px]), 2),
                words=[c["w"] for c in cand[:N_WORDS]],
                cand=cand,
            ))
        n_sea = int((basin == 0).sum())
        n_small = int(sum(counts[1:] < args.min_n))
        print(f"σ={sigma}: 峰 {len(peaks)}, 地名 {len(places)}（{args.min_n} 件未満の山域 {n_small} は無名）, "
              f"海 {n_sea:,} 件, 山域件数 中央値 {int(np.median(counts[1:])):,} / 最大 {int(counts[1:].max()):,}")
        levels.append(dict(sigma=sigma, min_n=args.min_n, n_peaks=len(peaks), places=places))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        coords=str(args.coords), grid=[nx, ny], extent=extent, n_awards=len(df),
        params=dict(peak_rel=PEAK_REL, mask_rel=MASK_REL, score_k=SCORE_K, alpha=ALPHA, gamma=GAMMA, min_frac=MIN_FRAC, min_count=MIN_COUNT,
                    n_words=N_WORDS, stop_words=sorted(STOP_WORDS)),
        levels=levels,
    ), ensure_ascii=False, indent=1), encoding="utf-8")
    basins_df.write_parquet(OUT_BASINS)
    print(f"→ {OUT_JSON} ({OUT_JSON.stat().st_size // 1024} KB), {OUT_BASINS}")


if __name__ == "__main__":
    main()
