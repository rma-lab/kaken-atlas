"""テキスト由来の連続色で UMAP 2D 地図を塗るプロトタイプ（A/B/D 比較）。

案A: 3D UMAP 座標を OKLab へ写像（完全教師なし）。
     3D 座標を PCA 回転し、分散の大きい2軸→(a,b)（色相・彩度平面）、残る1軸→L（明度、
     狭レンジに圧縮）。埋め込みで近い課題ほど知覚的に近い色になる。凡例は作れない。

案B: 大区分アンカー混色（半教師・解釈可能）。
     11大区分の埋め込み重心とのコサイン類似度を softmax 重みにして、意味順色相環の
     11色を OKLab 上で混色。単一分野に強く寄る課題はアンカー色、複数分野にまたがる
     課題は混色＝灰色化（彩度が学際性を自動エンコード）。
     softmax の逆温度 β は「top-1 重みの平均 = TARGET_TOP1」となるよう自動決定。
     混色後の彩度は 95%タイルがアンカー彩度(0.145)になるよう一括スケール。

案D: 弾性リング（閉じた主曲線＝768次元空間に張る「色相の輪ゴム」・ほぼ教師なし）。
     円環トポロジーのノード列（リングSOM）を埋め込み空間でEM学習して雲の密度に沿わせ、
     各ノードに色相角を割り当てる（初期化＝11重心の最短巡回路。ラベルは初期化にのみ使用）。
     各点の色は近傍ノードの色相単位ベクトルのカーネル重み付き平均：
     **向き＝色相、長さ（円環合意度 R）＝彩度**。リングの反対側から等距離の点（分野の狭間）は
     打ち消し合って灰色になる。カーネル幅 h は「R の中央値 = TARGET_R」となるよう自動決定。

出力: reports/figures/map_textcolor_A.png / map_textcolor_B.png / map_textcolor_D.png
学習したリングは data/interim/elastic_ring_nodes.npz に保存。

使い方:
    uv run python scripts/plot_map_textcolor.py [abd]   # 引数で対象を選択（既定: 全部）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import sys

sys.path.insert(0, "src")

UMAP2D = Path("data/processed/umap2d_nn15_md0.1.parquet")
UMAP3D = Path("data/processed/umap3d_nn15_md0.1.parquet")
EMBEDDINGS = Path("data/processed/embeddings.parquet")
OUTDIR = Path("reports/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"

SEED = 42
TARGET_TOP1 = 0.60  # 案B: softmax top-1 重みの平均をこの値に合わせて β を決める
ANCHOR_CHROMA = 0.145  # 意味順色相環のアンカー彩度（OKLCH C）

# 案D: 弾性リング
RING_NODES = 198  # 11セグメント × 18ノード
RING_ITERS = 60
RING_SIGMA0 = 24.0  # SOM近傍幅（ノード数単位）の初期値
RING_SIGMA1 = 1.5  # 同・最終値（幾何アニーリング）
TARGET_R = 0.60  # 円環合意度 R の中央値をこの値に合わせてカーネル幅 h を決める
CYCLE = list("AJCBDEKFGHI")  # 11重心の最短巡回路（意味順色相環の導出順）
RING_NPZ = Path("data/interim/elastic_ring_nodes.npz")

FOOTER_COMMON = (
    "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | 2019–2025年度開始の採択課題 206,078件"
    "（採択時概要あり・不採択除く）\n埋め込み: cl-nagoya/ruri-v3-310m（768次元）| "
    "UMAP (cosine, n_neighbors=15, seed=42) | 作成: KAKEN-ATLAS (26K15524)"
)


# ---------------------------------------------------------------- OKLab 変換

def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """OKLab (N,3) → sRGB (N,3)。ガマット外は範囲外の値のまま返す（クランプしない）。"""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    rgb_lin = np.stack(
        [
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        axis=1,
    )
    srgb = np.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * np.sign(rgb_lin) * np.abs(rgb_lin) ** (1 / 2.4) - 0.055,
    )
    return srgb


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """sRGB (N,3) → OKLab (N,3)。"""
    rgb_lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    r, g, b = rgb_lin[:, 0], rgb_lin[:, 1], rgb_lin[:, 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        ],
        axis=1,
    )


def gamut_clamp(lab: np.ndarray, iters: int = 16) -> np.ndarray:
    """L を保ったまま彩度 (a,b) を二分法で縮めて sRGB ガマット内に収める。"""
    lab = lab.copy()
    srgb = oklab_to_srgb(lab)
    out = ((srgb < 0) | (srgb > 1)).any(axis=1)
    if not out.any():
        return np.clip(srgb, 0, 1)
    lo = np.zeros(out.sum())
    hi = np.ones(out.sum())
    sub = lab[out]
    for _ in range(iters):
        mid = (lo + hi) / 2
        test = sub.copy()
        test[:, 1:] *= mid[:, None]
        bad = ((oklab_to_srgb(test) < 0) | (oklab_to_srgb(test) > 1)).any(axis=1)
        hi = np.where(bad, mid, hi)
        lo = np.where(bad, lo, mid)
    sub[:, 1:] *= lo[:, None]
    lab[out] = sub
    return np.clip(oklab_to_srgb(lab), 0, 1)


def hex_to_rgb01(hexcolor: str) -> np.ndarray:
    h = hexcolor.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)])


# ---------------------------------------------------------------- 案A

def colors_scheme_a() -> tuple[pl.DataFrame, str]:
    """3D UMAP → PCA回転 → OKLab。award_number と RGB を返す。"""
    df3 = pl.read_parquet(UMAP3D)
    xyz = df3.select("c0", "c1", "c2").to_numpy()
    xyz = xyz - xyz.mean(axis=0)
    # PCA 回転: 分散大→(a,b) 平面、分散小→L
    _, _, vt = np.linalg.svd(xyz, full_matrices=False)
    rot = xyz @ vt.T  # 列0,1,2 = 分散降順

    def robust_scale(v: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(v, [2, 98])
        return np.clip((v - (lo + hi) / 2) / ((hi - lo) / 2), -1, 1)

    a = robust_scale(rot[:, 0]) * 0.13
    b = robust_scale(rot[:, 1]) * 0.13
    L = 0.63 + robust_scale(rot[:, 2]) * 0.11  # L ∈ [0.52, 0.74]
    rgb = gamut_clamp(np.stack([L, a, b], axis=1))
    params = "色=3D UMAP座標→OKLab写像（PCA回転, 主2軸→(a,b)±0.13, 第3軸→L∈[0.52,0.74], ガマットは彩度圧縮で収容）"
    return df3.select("award_number").with_columns(
        r=pl.Series(rgb[:, 0]), g=pl.Series(rgb[:, 1]), b=pl.Series(rgb[:, 2])
    ), params


# ---------------------------------------------------------------- 案B

def colors_scheme_b() -> tuple[pl.DataFrame, str, dict]:
    """大区分アンカー混色。award_number と RGB、診断情報を返す。"""
    from kaken_atlas.kubun import DAI_COLORS, load_dai_labels

    emb_df = pl.read_parquet(EMBEDDINGS)
    emb = emb_df["embedding"].to_numpy()  # (N, 768) L2 正規化済み
    labels = emb_df.select("award_number").join(
        load_dai_labels(), on="award_number", how="left"
    )["dai"].to_numpy()

    # 11 大区分の重心（一意所属の課題のみで計算）→ 正規化
    keys = list(DAI_COLORS)
    cents = np.stack([emb[labels == k].mean(axis=0) for k in keys])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    sims = emb @ cents.T  # (N, 11) コサイン類似度

    # β 自動決定: top-1 重みの平均が TARGET_TOP1 になるよう二分法
    def mean_top1(beta: float) -> float:
        z = beta * (sims - sims.max(axis=1, keepdims=True))
        w = np.exp(z)
        w /= w.sum(axis=1, keepdims=True)
        return float(w.max(axis=1).mean())

    lo, hi = 1.0, 500.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if mean_top1(mid) < TARGET_TOP1:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    z = beta * (sims - sims.max(axis=1, keepdims=True))
    w = np.exp(z)
    w /= w.sum(axis=1, keepdims=True)

    # アンカー色を OKLab にして重み混色
    anchors_lab = srgb_to_oklab(np.stack([hex_to_rgb01(DAI_COLORS[k]) for k in keys]))
    lab = w @ anchors_lab  # (N, 3)

    # 彩度の一括スケール: 95%タイルをアンカー彩度に合わせる
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    gain = ANCHOR_CHROMA / np.percentile(chroma, 95)
    lab[:, 1:] *= gain
    rgb = gamut_clamp(lab)

    diag = {
        "beta": beta,
        "mean_top1": float(w.max(axis=1).mean()),
        "gain": gain,
        "chroma_med": float(np.median(chroma) * gain),
    }
    params = (
        f"色=大区分重心とのコサイン類似度をsoftmax混色（β={beta:.0f}: top-1重み平均{TARGET_TOP1:.2f}に自動調整, "
        f"OKLab混色, 彩度95%タイル={ANCHOR_CHROMA}に正規化）| 彩度低=複数分野の中間"
    )
    return emb_df.select("award_number").with_columns(
        r=pl.Series(rgb[:, 0]), g=pl.Series(rgb[:, 1]), b=pl.Series(rgb[:, 2])
    ), params, diag


# ---------------------------------------------------------------- 案D

def _fit_elastic_ring(emb: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """リングSOM（円環トポロジーのバッチSOM）を学習。ノード座標とノード色相角を返す。

    初期化: 11大区分重心の最短巡回路上にノードを等分配置し、色相角は意味順色相環の
    アンカー色相をセグメント内で円弧補間して割り当てる。学習中ノードの添字は不変なので
    色相はノードに貼り付いたまま雲に沿って移動するが、リングが雲に沿って滑るため色相の
    配分が区分ごとに伸縮する（実測: ±60°程度、全体回転では補正不能）。そこで学習後に
    **色合わせ**を行う: 各大区分の担当ノード添字の円環平均（中心ノード）を求め、そこに
    その区分のアンカー色相をピン留めし、中心間は順序を保って円弧を線形補間する単調な
    円環ワープで全ノードの色相を再割り当てする。凡例の11色と色相の対応が回復する代わり、
    「データの多い区分ほど広い色相域を得る」という学習結果の配分は放棄する。
    """
    from kaken_atlas.kubun import DAI_COLORS

    m = RING_NODES
    per_seg = m // len(CYCLE)
    assert per_seg * len(CYCLE) == m, "RING_NODES は 11 の倍数にすること"

    cents = np.stack([emb[labels == k].mean(axis=0) for k in CYCLE])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    anchor_lab = srgb_to_oklab(np.stack([hex_to_rgb01(DAI_COLORS[k]) for k in CYCLE]))
    anchor_hue = np.arctan2(anchor_lab[:, 2], anchor_lab[:, 1])

    nodes = np.empty((m, emb.shape[1]), dtype=np.float32)
    hues = np.empty(m)
    for s in range(len(CYCLE)):
        nxt = (s + 1) % len(CYCLE)
        t = np.arange(per_seg) / per_seg
        seg = (1 - t)[:, None] * cents[s] + t[:, None] * cents[nxt]
        nodes[s * per_seg : (s + 1) * per_seg] = seg / np.linalg.norm(
            seg, axis=1, keepdims=True
        )
        dh = (anchor_hue[nxt] - anchor_hue[s] + np.pi) % (2 * np.pi) - np.pi  # 最短円弧
        hues[s * per_seg : (s + 1) * per_seg] = anchor_hue[s] + t * dh

    # 円環インデックス距離の近傍カーネル行列を作る関数
    idx = np.arange(m)
    ring_d = np.minimum(np.abs(idx[:, None] - idx[None, :]), m - np.abs(idx[:, None] - idx[None, :]))

    from scipy import sparse

    n = emb.shape[0]
    for it in range(RING_ITERS):
        sigma = RING_SIGMA0 * (RING_SIGMA1 / RING_SIGMA0) ** (it / (RING_ITERS - 1))
        kmat = np.exp(-(ring_d**2) / (2 * sigma**2))
        bmu = (emb @ nodes.T).argmax(axis=1)
        onehot = sparse.csr_matrix(
            (np.ones(n, dtype=np.float32), (bmu, np.arange(n))), shape=(m, n)
        )
        sums = onehot @ emb  # (m, 768) BMUごとの合計
        counts = np.bincount(bmu, minlength=m).astype(np.float64)
        new = (kmat @ sums) / (kmat @ counts)[:, None]  # 円環近傍で平滑化した重心
        norms = np.linalg.norm(new, axis=1, keepdims=True)
        nodes = (new / norms).astype(np.float32)

    # 色合わせ: 各大区分の中心ノード（担当ノード添字の円環平均）にアンカー色相を
    # ピン留めし、中心間を順序保存の円弧線形補間で埋める単調な円環ワープ
    bmu = (emb @ nodes.T).argmax(axis=1)
    # 各区分の中心 = 初期セグメント中心を基準に ±1/4 周の窓内にある担当ノード添字の
    # 円環中央値（環上に散らばる低集中度の区分でも素朴な円環平均のように破綻しない）
    centers = np.empty(len(CYCLE))
    for s, k in enumerate(CYCLE):
        init_c = (s + 0.5) * per_seg
        diff = (bmu[labels == k] - init_c + m / 2) % m - m / 2  # [-m/2, m/2)
        win = diff[np.abs(diff) < m / 4]
        centers[s] = init_c + (np.median(win) if win.size else 0.0)

    # 単調性の強制（centers は初期セグメント基準で展開済み。交差する隣接ペアを
    # 平均に寄せて解消、最小間隔1ノード）
    unwrapped = centers.copy()
    for _ in range(50):
        bad = np.diff(unwrapped) < 1.0
        if not bad.any():
            break
        for i in np.where(bad)[0]:
            mid = (unwrapped[i] + unwrapped[i + 1]) / 2
            unwrapped[i], unwrapped[i + 1] = mid - 0.5, mid + 0.5
    seg_len = np.diff(np.append(unwrapped, unwrapped[0] + m))
    if (seg_len <= 0).any():
        print("警告: 区分中心の円環順を単調化できませんでした")
    unwrapped = np.append(unwrapped, unwrapped[0] + m)  # 長さ12

    # 各ノードの位置を区分中心に対する割合に変換し、アンカー色相を円弧補間
    dh = (np.diff(np.append(anchor_hue, anchor_hue[0])) + np.pi) % (2 * np.pi) - np.pi
    base = unwrapped[0]
    pos = (np.arange(m) - base) % m  # 先頭区分の中心を起点にした周回位置
    seg_idx = np.searchsorted(unwrapped - base, pos, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(CYCLE) - 1)
    t = (pos - (unwrapped[seg_idx] - base)) / seg_len[seg_idx]
    hues = anchor_hue[seg_idx] + t * dh[seg_idx]

    # 診断: ワープ後の区分ごと残差（点の受け取る色相の円環平均 vs アンカー）
    point_hue = hues[bmu]
    resid = []
    for k, ah in zip(CYCLE, anchor_hue, strict=True):
        mask = labels == k
        mu = np.arctan2(np.sin(point_hue[mask]).mean(), np.cos(point_hue[mask]).mean())
        resid.append(np.degrees((ah - mu + np.pi) % (2 * np.pi) - np.pi))
    return nodes, hues, float(np.max(np.abs(resid)))


def colors_scheme_d() -> tuple[pl.DataFrame, str, dict, tuple]:
    """弾性リング色。award_number+RGB、脚注、診断、地図重畳用ノード情報を返す。"""
    emb_df = pl.read_parquet(EMBEDDINGS)
    emb = emb_df["embedding"].to_numpy()
    from kaken_atlas.kubun import load_dai_labels

    labels = emb_df.select("award_number").join(
        load_dai_labels(), on="award_number", how="left"
    )["dai"].to_numpy()

    nodes, hues, max_resid = _fit_elastic_ring(emb, labels)
    RING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(RING_NPZ, nodes=nodes, hues=hues)

    sims = emb @ nodes.T
    d2 = np.maximum(2.0 - 2.0 * sims, 0.0)  # 単位球上の二乗ユークリッド距離
    d2 -= d2.min(axis=1, keepdims=True)
    ux, uy = np.cos(hues), np.sin(hues)

    def resultant(h: float, sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = np.exp(-sub / h)
        w /= w.sum(axis=1, keepdims=True)
        vx, vy = w @ ux, w @ uy
        return np.hypot(vx, vy), np.arctan2(vy, vx)

    # h 自動決定: R の中央値 = TARGET_R（2万点のサブサンプルで二分法、h小→R→1 で単調減少）
    rng = np.random.default_rng(SEED)
    sub = d2[rng.choice(len(d2), 20_000, replace=False)]
    lo, hi = 1e-4, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        r_med = float(np.median(resultant(mid, sub)[0]))
        if r_med > TARGET_R:
            lo = mid
        else:
            hi = mid
    h = (lo + hi) / 2
    r, hue = resultant(h, d2)

    # 彩度 = R を案Bと同じ規則で正規化（95%タイル→アンカー彩度）
    gain = ANCHOR_CHROMA / np.percentile(r, 95)
    chroma = r * gain
    lab = np.stack([np.full_like(chroma, 0.60), chroma * np.cos(hue), chroma * np.sin(hue)], axis=1)
    rgb = gamut_clamp(lab)

    # 地図重畳用: 各ノードの担当点（BMU）の 2D 座標平均とノード色
    bmu = sims.argmax(axis=1)
    umap2d = pl.read_parquet(UMAP2D)
    xy = (
        emb_df.select("award_number")
        .join(umap2d, on="award_number", how="left")
        .select("c0", "c1")
        .to_numpy()
    )
    counts = np.bincount(bmu, minlength=RING_NODES)
    node_x = np.bincount(bmu, weights=xy[:, 0], minlength=RING_NODES)
    node_y = np.bincount(bmu, weights=xy[:, 1], minlength=RING_NODES)
    ok = counts > 0
    node_x, node_y = node_x[ok] / counts[ok], node_y[ok] / counts[ok]
    node_lab = np.stack(
        [np.full(ok.sum(), 0.60), ANCHOR_CHROMA * np.cos(hues[ok]), ANCHOR_CHROMA * np.sin(hues[ok])],
        axis=1,
    )
    overlay = (node_x, node_y, gamut_clamp(node_lab))

    diag = {
        "h": h,
        "r_med": float(np.median(r)),
        "gain": gain,
        "bmu_cos_mean": float(sims.max(axis=1).mean()),
        "empty_nodes": int((~ok).sum()),
        "max_resid_deg": max_resid,
    }
    params = (
        f"色相=768次元に張った弾性リング（{RING_NODES}ノード, リングSOM {RING_ITERS}反復, "
        f"σ={RING_SIGMA0:.0f}→{RING_SIGMA1}, 初期化=11重心巡回路, "
        f"色合わせ=区分中心を11アンカー色相にピン留めする円環ワープ）| "
        f"彩度=円環合意度R（カーネル幅h={h:.3f}: R中央値{TARGET_R}に自動調整, 95%タイル={ANCHOR_CHROMA}に正規化）"
        "| 灰色=リング反対側と等距離＝分野の狭間 | ○=リングの通り道（担当課題の平均位置）"
    )
    return emb_df.select("award_number").with_columns(
        r=pl.Series(rgb[:, 0]), g=pl.Series(rgb[:, 1]), b=pl.Series(rgb[:, 2])
    ), params, diag, overlay


# ---------------------------------------------------------------- 描画

def plot(colors: pl.DataFrame, title: str, params: str, out: Path, legend: bool,
         overlay: tuple | None = None) -> None:
    df = pl.read_parquet(UMAP2D).join(colors, on="award_number", how="inner")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(df.height)  # 描画順バイアスを避ける
    x = df["c0"].to_numpy()[order]
    y = df["c1"].to_numpy()[order]
    rgb = df.select("r", "g", "b").to_numpy()[order]

    fig, ax = plt.subplots(figsize=(12, 11), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(x, y, s=0.35, c=rgb, alpha=0.55, linewidths=0, rasterized=True)
    if overlay is not None:
        nx, ny, nrgb = overlay
        ax.plot(np.append(nx, nx[0]), np.append(ny, ny[0]),
                color="#00000030", linewidth=0.8, zorder=3)
        ax.scatter(nx, ny, s=22, c=nrgb, edgecolors="white", linewidths=0.7, zorder=4)
    if legend:
        from kaken_atlas.kubun import DAI_COLORS, DAI_GLOSS

        handles = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                       markerfacecolor=c, markeredgecolor="none",
                       label=f"{k}〈{DAI_GLOSS[k]}〉")
            for k, c in DAI_COLORS.items()
        ]
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", markersize=7,
                       markerfacecolor="#9a9a94", markeredgecolor="none",
                       label="灰色寄り＝複数分野の中間")
        )
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize=9, frameon=False, labelcolor=INK,
                  title="アンカー色（大区分重心）", title_fontsize=10)
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    ax.annotate(
        FOOTER_COMMON + "\n" + params,
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
    OUTDIR.mkdir(parents=True, exist_ok=True)
    schemes = "".join(sys.argv[1:]).lower() or "abd"

    if "a" in schemes:
        colors_a, params_a = colors_scheme_a()
        plot(
            colors_a,
            "学術地図 × テキスト由来色 案A（3D UMAP → OKLab 直接写像・完全教師なし）",
            params_a,
            OUTDIR / "map_textcolor_A.png",
            legend=False,
        )

    if "b" in schemes:
        colors_b, params_b, diag = colors_scheme_b()
        print(f"案B診断: β={diag['beta']:.1f}, top-1平均={diag['mean_top1']:.3f}, "
              f"彩度gain={diag['gain']:.2f}, 彩度中央値={diag['chroma_med']:.3f}")
        plot(
            colors_b,
            "学術地図 × テキスト由来色 案B（大区分アンカー混色・彩度=単一分野への集中度）",
            params_b,
            OUTDIR / "map_textcolor_B.png",
            legend=True,
        )

    if "d" in schemes:
        colors_d, params_d, diag_d, overlay = colors_scheme_d()
        print(f"案D診断: h={diag_d['h']:.4f}, R中央値={diag_d['r_med']:.3f}, "
              f"彩度gain={diag_d['gain']:.2f}, BMU平均cos={diag_d['bmu_cos_mean']:.3f}, "
              f"空ノード={diag_d['empty_nodes']}, 色合わせ後の最大残差={diag_d['max_resid_deg']:.1f}°")
        plot(
            colors_d,
            "学術地図 × テキスト由来色 案D（弾性リング＝色相の輪ゴム・彩度=円環合意度）",
            params_d,
            OUTDIR / "map_textcolor_D.png",
            legend=False,
            overlay=overlay,
        )


if __name__ == "__main__":
    main()
