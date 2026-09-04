"""固定UMAPレイアウト上の年度別KDE — 時系列プロトタイプ（粘性流体構想の予告編）。

設計制約（重要）: レイアウトは全期間一括UMAPで固定し、開始年度ごとの点集合で
濃度だけを変化させる（年別にUMAPを再実行すると配置が変わり動画にならない）。

年度ごとの採択数変動（24.7k〜36.3k件）に引きずられないよう、各年度の密度は
その年度の件数で正規化した「シェア密度」で描く。カラースケールは全年度共通。

出力（reports/figures/）:
1. kde_years_facets.png — 年度別7パネルの静止図（スライド用）
2. kde_anomaly.mp4      — 偏差の推移動画（各年度シェア−全期間平均、赤=超過/青=未満）
   密度場は年々ほぼ不変のため、動画は密度そのものではなく偏差で見せる
   （密度そのものの動画は色をどう変えても動かず不採用。擬似カラー版も試作の上で廃止）
2'. kde_drift2019.mp4   — 同じ仕組みで基準を2019年度にした累積変化版（真っ白から始まり
    有意な変化だけが浮かんで年々育つ。デモ本命）
3. kde_change.png       — 後期(2023–2025)−前期(2019–2021)のシェア差分マップ（横長・
   成長領域3箇所を図中ラベル: 量子技術/生態学・環境/医療リアルワールドデータ）
4. kde_genai.png        — 生成AI関連キーワード課題の分布図（横長・密集地3箇所を図中ラベル:
   教育工学/自然言語処理/医療・看護情報学。「2024出芽」リトマス試験の図）
※動画2本は|偏差|>2σの有意性マスク付き（σ=年度内半分割ブートストラップによる画素別推定）
※クロスフェードは見やすさのための演出であり、年度の中間状態の推定ではない

使い方:
    uv run python scripts/plot_kde_years.py [data/processed/umap2d_nn15_md0.1.parquet]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Hiragino Sans"  # macOS の日本語フォント
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

# dataviz 基準パレット（plot_map.py と同一）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG = "#2a78d6"  # 減少（青系）
DIV_POS = "#b2532e"  # 増加（赤系）— ユーザ指定: 赤=増加/青=減少のイメージ

YEARS = list(range(2019, 2026))
GRID = 640           # 濃度場の格子解像度（長辺ピクセル数）
# ガウス平滑の帯域幅（UMAP座標系の単位）。0.15は斑点状、0.8は輪郭消失（σ比較検証 2026-08-30）
SIGMA_DATA = 0.3
SIGMA_CHANGE = 0.5   # 差分マップ用。差分はノイズが増幅されるため一段粗くする
MARGIN = 0.5         # 描画範囲の余白（同上）

# 生成AI リトマス試験のキーワード（タイトル/キーワード/概要の連結 text 列に対する部分一致）
GENAI_KEYWORDS = ["生成AI", "生成系AI", "大規模言語モデル", "ChatGPT", "GPT-4", "LLM"]

# 出所・パラメータの脚注（横長図では1行ずつ縦に積む）
FOOTNOTE_LINES = [
    "データ: KAKEN科研費データベース（国立情報学研究所）より取得",
    "対象: 2019–2025年度開始の採択課題 206,078件（採択時概要あり・不採択除く）",
    "埋め込み: cl-nagoya/ruri-v3-310m（768次元）",
    "次元削減: UMAP (cosine, n_neighbors=15, seed=42, 全期間一括・レイアウト固定)",
    f"密度: 格子ヒストグラム＋ガウス平滑（σ={SIGMA_DATA} 座標単位）を"
    "各年度の件数で正規化（シェア密度）",
    "作成: KAKEN-ATLAS (26K15524)",
]
FOOTNOTE = (
    f"{FOOTNOTE_LINES[0]} | {FOOTNOTE_LINES[1]}\n"
    f"{FOOTNOTE_LINES[2]} | {FOOTNOTE_LINES[3]}\n"
    f"{FOOTNOTE_LINES[4]} | {FOOTNOTE_LINES[5]}"
)


def load(coords_path: Path) -> pl.DataFrame:
    coords = pl.read_parquet(coords_path)
    corpus = pl.read_parquet("data/processed/corpus.parquet").select(
        ["award_number", "start_fy", "text"]
    )
    return coords.join(corpus, on="award_number", how="left")


def density_fields(
    df: pl.DataFrame, sigma: float = SIGMA_DATA
) -> tuple[dict[int, np.ndarray], list[float], dict[int, int]]:
    """年度ごとのシェア密度場と描画範囲・件数を返す。"""
    x, y = df["c0"].to_numpy(), df["c1"].to_numpy()
    x0, x1 = x.min() - MARGIN, x.max() + MARGIN
    y0, y1 = y.min() - MARGIN, y.max() + MARGIN
    nx = GRID
    ny = int(round(GRID * (y1 - y0) / (x1 - x0)))
    px_per_unit = nx / (x1 - x0)
    sigma_px = sigma * px_per_unit

    fields: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    fy = df["start_fy"].to_numpy()
    for year in YEARS:
        m = fy == year
        counts[year] = int(m.sum())
        h, _, _ = np.histogram2d(
            x[m], y[m], bins=[nx, ny], range=[[x0, x1], [y0, y1]]
        )
        # シェア正規化: その年度の1件が持つ質量を 1/n に揃える
        fields[year] = gaussian_filter(h.T / counts[year], sigma=sigma_px)
    return fields, [x0, x1, y0, y1], counts


def noise_std_fields(
    df: pl.DataFrame, sigma: float, n_splits: int = 4, seed: int = 42
) -> dict[int, np.ndarray]:
    """年度ごとのシェア密度場の標本ノイズ（画素別標準偏差）を半分割ブートストラップで推定。

    年度内の点を半々に割った2つの場の差は Var(a-b)=4·Var(フル年場) を満たす
    （半分サイズの場の分散はフル場の2倍）ので、E[(a-b)²]/4 がフル年場の分散推定。
    """
    x, y = df["c0"].to_numpy(), df["c1"].to_numpy()
    x0, x1 = x.min() - MARGIN, x.max() + MARGIN
    y0, y1 = y.min() - MARGIN, y.max() + MARGIN
    nx = GRID
    ny = int(round(GRID * (y1 - y0) / (x1 - x0)))
    sigma_px = sigma * nx / (x1 - x0)

    fy = df["start_fy"].to_numpy()
    out: dict[int, np.ndarray] = {}
    for year in YEARS:
        idx = np.where(fy == year)[0]
        sq = np.zeros((ny, nx))
        for s in range(n_splits):
            rng = np.random.default_rng(seed + s)
            perm = rng.permutation(idx)
            half = len(perm) // 2
            fa_fb = []
            for part in (perm[:half], perm[half:]):
                h, _, _ = np.histogram2d(
                    x[part], y[part], bins=[nx, ny], range=[[x0, x1], [y0, y1]]
                )
                fa_fb.append(gaussian_filter(h.T / len(part), sigma=sigma_px))
            sq += (fa_fb[0] - fa_fb[1]) ** 2
        out[year] = np.sqrt(sq / n_splits / 4)
    return out


def _clean_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_BLUE)


def plot_facets(
    fields: dict[int, np.ndarray], extent: list[float], counts: dict[int, int], out: Path
) -> None:
    vmax = max(np.quantile(f, 0.999) for f in fields.values())
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.8), facecolor=SURFACE)
    for ax, year in zip(axes.flat, YEARS, strict=False):
        ax.set_facecolor(SURFACE)
        ax.imshow(
            fields[year], origin="lower", extent=extent, cmap=_cmap(), norm=norm,
            interpolation="bilinear",
        )
        ax.set_title(f"{year}年度（{counts[year]:,}件）", color=INK, fontsize=11)
        _clean_axes(ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.05, 0.6,
        "各年度に開始した採択課題の\nシェア密度（件数正規化）\nカラースケールは全年度共通",
        color=MUTED, fontsize=10, va="top",
    )
    fig.suptitle(
        "科研費 採択課題の意味空間 — 年度別密度（レイアウト固定）",
        color=INK, fontsize=15, y=0.98,
    )
    fig.text(0.01, -0.01, FOOTNOTE, color=MUTED, fontsize=7.5)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def _year_frames(
    fields: dict[int, np.ndarray], hold: int, blend: int
) -> list[tuple[np.ndarray, str]]:
    """年度ごとの静止＋クロスフェード遷移のフレーム列を組む。"""
    frames: list[tuple[np.ndarray, str]] = []
    for i, year in enumerate(YEARS):
        frames += [(fields[year], f"{year}年度")] * hold
        if i < len(YEARS) - 1:
            nxt = YEARS[i + 1]
            for t in np.linspace(0, 1, blend + 2)[1:-1]:
                frames.append(
                    ((1 - t) * fields[year] + t * fields[nxt], f"{year}→{nxt}")
                )
    frames += [(fields[YEARS[-1]], f"{YEARS[-1]}年度")] * hold
    return frames


def _save_movie(fig, im, label, frames, fps: int, out: Path) -> None:
    def update(i: int):
        im.set_data(frames[i][0])
        label.set_text(frames[i][1])
        return im, label

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000))
    plt.close(fig)
    print(f"出力: {out}（{len(frames)}フレーム, {len(frames) / fps:.0f}秒）")


def make_anomaly_movie(
    fields: dict[int, np.ndarray], extent: list[float], out: Path,
    baseline: str = "mean",
    noise_std: dict[int, np.ndarray] | None = None,
    z_thresh: float = 2.0,
) -> None:
    """変化を主役にした動画: 各年度シェア − 基準シェア（赤=超過, 青=未満）。

    密度場は年々ほぼ不変で、密度そのものの動画では変化が視認できない。
    偏差を色にすることで「その年どこが厚かったか」だけが動く。
    全期間平均の等高線（灰色）を敷いて地形の目印にする。

    baseline="mean": 全期間平均との偏差（その年らしさの天気図。一過性パルスが往復で見える）
    baseline="first": 初年度(2019)との差（累積ドリフト。トレンドが年々濃くなる。
        基準が単年サンプルのため2019年の標本ノイズが全フレームに逆符号で乗る点に注意）
    """
    mean = np.mean(list(fields.values()), axis=0)  # 等高線は常に全期間平均の地形
    base = fields[YEARS[0]] if baseline == "first" else mean
    anomalies = {y: fields[y] - base for y in YEARS}

    # 有意性マスク: 標本ノイズ2σ以下の偏差は白のまま（σ=0.5では年次揺らぎの大半が
    # 標本ノイズと同規模であることが半分割ブートストラップで判明したため。2026-08-30）
    if noise_std is not None:
        if baseline == "first":
            var_base = noise_std[YEARS[0]] ** 2
        else:
            var_base = np.sum([noise_std[y] ** 2 for y in YEARS], axis=0) / len(YEARS) ** 2
        for y in YEARS:
            sd = np.sqrt(noise_std[y] ** 2 + var_base)
            anomalies[y] = np.where(np.abs(anomalies[y]) > z_thresh * sd, anomalies[y], 0.0)
    lim = max(np.quantile(np.abs(a), 0.999) for a in anomalies.values())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, SURFACE, DIV_POS])
    frames = _year_frames(anomalies, hold=5, blend=12)

    fig, ax = plt.subplots(figsize=(9, 9.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(
        frames[0][0], origin="lower", extent=extent, cmap=cmap, norm=norm,
        interpolation="bilinear",
    )
    levels = np.quantile(mean[mean > mean.max() * 1e-3], [0.35, 0.65, 0.88])
    ny, nx = mean.shape
    gx = np.linspace(extent[0], extent[1], nx)
    gy = np.linspace(extent[2], extent[3], ny)
    ax.contour(gx, gy, mean, levels=levels, colors="#c3c2b7", linewidths=0.6)
    label = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, color=INK, fontsize=22,
        fontweight="bold", va="top",
    )
    if baseline == "first":
        title = f"研究地形の変化 — 各年度シェア − {YEARS[0]}年度シェア（赤=増加, 青=減少）"
        base_note = (
            f"色: 各年度のシェア密度 − {YEARS[0]}年度シェア密度"
            f"（累積変化, 平滑幅σ={SIGMA_CHANGE}）"
        )
    else:
        title = "研究地形の年度別偏差 — 各年度シェア − 全期間平均（赤=超過, 青=未満）"
        base_note = f"色: 各年度のシェア密度 − 2019–2025平均シェア密度（平滑幅σ={SIGMA_CHANGE}）"
    if noise_std is not None:
        base_note += (
            f" | 表示は|偏差|>{z_thresh:g}σの画素のみ"
            "（σ=年度内半分割ブートストラップによる標本ノイズの画素別推定）"
        )
    ax.set_title(title, color=INK, fontsize=13, pad=12)
    _clean_axes(ax)
    ax.annotate(
        FOOTNOTE
        + "\n" + base_note
        + " | 灰色等高線: 全期間平均密度（地形の目印）"
        + "\n年度間の遷移はクロスフェード（見やすさのための演出であり中間状態の推定ではない）",
        xy=(0, -0.04), xycoords="axes fraction", color=MUTED, fontsize=7,
    )
    _save_movie(fig, im, label, frames, fps=12, out=out)


def _landscape(title: str) -> tuple:
    """横長レイアウト: 左=地図、右=説明・脚注のテキスト列。キャプションと地図を分離する。"""
    fig = plt.figure(figsize=(16, 8), facecolor=SURFACE)
    ax = fig.add_axes([0.01, 0.03, 0.52, 0.88])
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.02, y=0.965, ha="left", color=INK, fontsize=15)
    return fig, ax


def _map_label(ax, xy, xytext, text, ha="left") -> None:
    """地図上の領域ラベル（白フチ付き・引き出し線）。"""
    ax.annotate(
        text, xy=xy, xytext=xytext, ha=ha, color=INK, fontsize=12, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=3, shrinkB=3),
        path_effects=[pe.withStroke(linewidth=3.5, foreground=SURFACE)],
    )


def plot_change(fields: dict[int, np.ndarray], extent: list[float], out: Path) -> None:
    """後期(2023–25)−前期(2019–21)のシェア差分。主要な成長領域を図中に直接ラベル。"""
    early = np.mean([fields[y] for y in (2019, 2020, 2021)], axis=0)
    late = np.mean([fields[y] for y in (2023, 2024, 2025)], axis=0)
    diff = late - early
    lim = np.quantile(np.abs(diff), 0.999)

    fig, ax = _landscape("研究地形の変化: 2023–2025年度シェア − 2019–2021年度シェア")
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, SURFACE, DIV_POS])
    im = ax.imshow(
        diff, origin="lower", extent=extent, cmap=cmap,
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), interpolation="bilinear",
    )
    _clean_axes(ax)
    cax = fig.add_axes([0.545, 0.20, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("シェア密度の差（赤=増加, 青=減少）", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    # 赤ピーク3箇所の領域名（周辺半径0.4の課題キーワード実査で同定 2026-08-30）
    _map_label(ax, (0.25, 1.02), (-5.0, 2.9), "量子技術・量子情報")
    _map_label(ax, (1.96, 5.77), (-5.0, 8.2), "生態学・環境\n（気候変動）")
    _map_label(ax, (5.00, 11.37), (7.4, 13.9), "医療リアルワールドデータ")

    fig.text(
        0.62, 0.90,
        "主な成長領域（赤のピーク）と周辺課題数の変化\n"
        "（ピーク半径0.4内・前期2019–21年 → 後期2023–25年）\n\n"
        "  量子技術・量子情報　　　　　 223 → 372件（+67%）\n"
        "  生態学・環境（気候変動）　　 356 → 435件（+22%）\n"
        "  医療リアルワールドデータ　　 273 → 382件（+40%）\n\n"
        "件数は直感のための生の値。前期・後期の総数がほぼ同じ\n"
        "（88,990件 vs 89,733件, +0.8%）ため件数の伸び率≒シェアの伸び率。\n"
        "領域名は各ピーク周辺課題のキーワード・タイトルの実査により同定。\n"
        "生成AI関連の伸びは帯状に分散するため点ピークとしては現れにくい\n"
        "（分布は別図 kde_genai.png を参照）。",
        va="top", color=INK, fontsize=13, linespacing=1.6,
    )
    fig.text(
        0.62, 0.32,
        "\n".join(
            FOOTNOTE_LINES
            + [
                f"差分の平滑幅はσ={SIGMA_CHANGE}"
                "（差分はノイズが増幅されるため密度図より粗くしている）"
            ]
        ),
        va="top", color=MUTED, fontsize=7.5, linespacing=1.6,
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def plot_genai(
    df: pl.DataFrame, fields: dict[int, np.ndarray], extent: list[float], out: Path
) -> None:
    """生成AI関連キーワードを含む課題の分布図。3つの密集地に分野名を直接ラベル。"""
    genai = df.filter(
        pl.any_horizontal(pl.col("text").str.contains(k, literal=True) for k in GENAI_KEYWORDS)
    )
    print("生成AI関連（キーワード部分一致）の年度別件数:")
    print(genai["start_fy"].value_counts().sort("start_fy"))

    fig, ax = _landscape(
        f"生成AI関連キーワードを含む課題の分布（2019–2025年度・{genai.height:,}件）"
    )
    # 背景: 全期間平均密度をごく薄いグレーで敷き、地形の目印にする
    mean = np.mean(list(fields.values()), axis=0)
    bg = LinearSegmentedColormap.from_list("bg", [SURFACE, "#cfcec5"])
    ax.imshow(
        mean, origin="lower", extent=extent, cmap=bg,
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=np.quantile(mean, 0.999)),
        interpolation="bilinear",
    )
    ax.scatter(genai["c0"], genai["c1"], s=4, c=INK, alpha=0.45, linewidths=0)
    _clean_axes(ax)

    # 密集地3箇所の分野名（周辺の生成AI課題のキーワード実査で同定 2026-08-30）
    _map_label(ax, (9.02, 9.72), (10.9, 12.6), "教育工学・学習支援", ha="right")
    _map_label(ax, (5.43, 5.86), (8.2, 3.0), "自然言語処理・AI研究")
    _map_label(ax, (5.43, 11.10), (0.2, 14.0), "医療・看護情報学")

    fig.text(
        0.60, 0.90,
        "黒点: 生成AI／生成系AI／大規模言語モデル／ChatGPT／GPT-4／LLM の\n"
        "いずれかをタイトル・キーワード・概要に含む課題。\n\n"
        "密集地3箇所（点密度ピーク・周辺課題の実査による同定）:\n"
        "  教育工学・学習支援　　教材開発・ChatGPT活用教育など\n"
        "  自然言語処理・AI研究　LLM本体・マルチモーダル基盤など\n"
        "  医療・看護情報学　　　電子カルテ・診断レポートのNLPなど\n\n"
        "LLM本体の研究（約100件）より教育・医療への応用が多く、\n"
        "技術の生産地と消費地が意味空間上で分離している。\n"
        "年度別件数は 24→40→58→98→104→307→366（2024年度に急増。\n"
        "2024年度課題の申請書は2023年秋＝ChatGPT公開後に執筆）。",
        va="top", color=INK, fontsize=13, linespacing=1.6,
    )
    fig.text(
        0.60, 0.30,
        "\n".join(FOOTNOTE_LINES + ["背景の灰色: 全期間平均のシェア密度（地形の目印）"]),
        va="top", color=MUTED, fontsize=7.5, linespacing=1.6,
    )
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"出力: {out}")


def main() -> None:
    default_coords = "data/processed/umap2d_nn15_md0.1.parquet"
    coords_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_coords)
    df = load(coords_path)
    fields, extent, counts = density_fields(df, SIGMA_DATA)
    fields_change, _, _ = density_fields(df, SIGMA_CHANGE)
    outdir = Path("reports/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plot_facets(fields, extent, counts, outdir / "kde_years_facets.png")
    plot_change(fields_change, extent, outdir / "kde_change.png")
    plot_genai(df, fields, extent, outdir / "kde_genai.png")
    # 偏差は差分と同じくノイズが増幅されるため σ=SIGMA_CHANGE の場を使う
    noise = noise_std_fields(df, SIGMA_CHANGE)
    make_anomaly_movie(
        fields_change, extent, outdir / "kde_anomaly.mp4", baseline="mean", noise_std=noise
    )
    make_anomaly_movie(
        fields_change, extent, outdir / "kde_drift2019.mp4", baseline="first", noise_std=noise
    )


if __name__ == "__main__":
    main()
