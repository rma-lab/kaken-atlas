"""ウェブ公開用の軽量インタラクティブ地図を生成する（GitHub Pages 向け）。

reports/figures/map_*_interactive.html（自己完結・約128MB）の外部データ版。
二段階読み込みで初期表示を数秒にする:

  フェーズ1  points.bin … 量子化座標(int16)＋gid→uid 対応表(uint32)。プログレスバー付きで取得→描画
  フェーズ2  ../shards/NNN.json … タイトル・キーワード等（SHARD_SIZE件/片、課題番号順＝uid 順、3 ビュー共有）。
             描画後に背景先読み（完了で検索が有効化）。未取得片への
             ホバーはその片だけ即時取得して穴埋めする。

トレース構成・配色・UI（検索/種目フィルタ/なげなわ集計/ツールチップ/KAKENリンク）は
plot_map_interactive.py と同一仕様。kaken_id は「KAKENHI-<種別>-<課題番号>」に
分解できるため種別コードのみ持つ。

HTML/CSS/JS の本体は scripts/web/index.template.html、Service Worker は scripts/web/sw.template.js（置換記号を埋めて出力）。

使い方:
    uv run python scripts/build_web_map.py data/processed/umap2d_nn15_md0.1.parquet
    uv run python scripts/build_web_map.py data/processed/umap3d_nn15_md0.1.parquet
    uv run python scripts/build_web_map.py <parquet> <出力先>   # 比較実験用（既定は docs/ 以下）
出力: docs/map2d/ または docs/map3d/（index.html + points.bin）と docs/shards/（3 ビュー共有・どのビューを生成しても同一内容）
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from plot_map_interactive import CATEGORY_ORDER, FOOTER  # noqa: E402

from kaken_atlas.kubun import DAI_COLORS, DAI_GLOSS, load_dai_labels  # noqa: E402

SHARD_SIZE = 2048  # 2の冪であること（JS側でビットシフトに使う）
# 球面ビュー: 点の半径の散らし（σ, クリップ）と、裏側を隠す不透明球の半径。点と球の隙間が小さいと、縮小時に
# 深度バッファの分解能が足りず z-fighting の縞が出る（2026-09-10 ユーザ報告）。隙間 ≥ 0.02 を確保する
GLOBE_JITTER_SIGMA = float(os.environ.get("GLOBE_JITTER_SIGMA", "0.002"))
GLOBE_JITTER_CLIP = float(os.environ.get("GLOBE_JITTER_CLIP", "0.005"))
GLOBE_SPHERE_R = float(os.environ.get("GLOBE_SPHERE_R", "0.985"))  # 既定の視距離での半径。縮小時は JS 側で距離に応じて小さくする
# 点の色（2026-09-11 決定）: 既定 "text"=研究内容から導いた連続色（弾性リング。scripts/compute_textcolor.py の色表を
# points.bin 末尾に RGB 各 1 バイトで同梱し、点ごとに塗る）。"dai"=従来の大区分 11 色（トレース単色。比較・実験用）。
# text のとき大区分の凡例・シートの色見本は「所属課題の平均色」（textcolor_legend.json）、カードの縁と見出しは点自身の色
POINT_COLOR = os.environ.get("POINT_COLOR", "text")
TEXTCOLOR_PARQUET = Path(os.environ.get("TEXTCOLOR_PARQUET", "data/processed/textcolor_d.parquet"))
TEXTCOLOR_LEGEND = TEXTCOLOR_PARQUET.with_name("textcolor_legend.json")
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def build_order(df: pl.DataFrame, parts: int = 1, merge_categories: bool = False) -> tuple[pl.DataFrame, list[dict]]:
    """plot_map_interactive.py と同じ描画順に並べ、トレース表を作る。

    点は「大区分（下層→上層）×種目（件数降順）」でトレースごとに連続配置。
    グローバル添字＝この並びの行番号がシャード分割の単位になる。

    parts > 1 のときは各点を乱数で parts 個の組に分け、「組0の全トレース → 組1の全トレース → …」の
    順に並べる（球面ビュー用）。半透明の点は描画順が後のトレースほど重なりの上に来て色が偏るが、
    組を多くして組ごとに大区分の順序を回転させると、ある画素の最上層の点はほぼランダムな1点になり、
    偏りが平均化される（組数は画素あたりの重なり数より多くする）。

    merge_categories=True のときは種目ごとに分けず「大区分×組」で1トレースにする（球面ビュー用）。
    Plotly の 3D はホバー/クリックの判定用バッファがオブジェクト番号を 8 ビットで持ち、
    **255 個を超えるトレースは判定できなくなる**ため、球面は 13大区分×16組=208 本に抑える。
    種目は点ごとの情報（シャードの行）として持たせ、種目フィルタは球面では使わない。
    """
    draw_order = ["区分なし", "複数", *DAI_COLORS.keys()]
    legend_order = [*DAI_COLORS.keys(), "複数", "区分なし"]
    if parts > 1:  # 再現性のため seed 固定
        rng = np.random.default_rng(7)
        df = df.with_columns(pl.Series("_part", rng.integers(0, parts, df.height)))
    else:
        df = df.with_columns(pl.lit(0).alias("_part"))
    parts_list: list[pl.DataFrame] = []
    traces: list[dict] = []
    offset = 0
    for dai in draw_order:  # 凡例アンカーは大区分ごとに1本（描画順の先頭側に置く）
        dsub = df.filter(pl.col("dai") == dai)
        if dsub.height == 0:
            continue
        color = DAI_COLORS.get(dai, "#cfcec7" if dai == "区分なし" else "#b9b8b0")
        label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
        traces.append(dict(
            k="a", dai=dai, label=label, color=color, n=dsub.height,
            rank=legend_order.index(dai) + 1, vis=True,  # 区分なしも既定で表示（2026-09-11 ユーザ判断。最下層に描くので色を覆わない）
        ))
    for part in range(parts):
        # 組ごとに大区分の順序を回転させ、「常に最後に描かれる大区分」を作らない
        order = draw_order[part % len(draw_order):] + draw_order[:part % len(draw_order)] if parts > 1 else draw_order
        for dai in order:
            dsub = df.filter((pl.col("dai") == dai) & (pl.col("_part") == part))
            if dsub.height == 0:
                continue
            color = DAI_COLORS.get(dai, "#cfcec7" if dai == "区分なし" else "#b9b8b0")
            label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
            visible = True
            # len 同数の種目間の順序を固定するため category 名でタイブレーク（出力の再現性）
            cat_counts = dsub.group_by("category").len().sort(
                ["len", "category"], descending=[True, False]
            )
            if merge_categories:  # 種目順に並べたうえで1トレースに
                subs = [dsub.filter(pl.col("category") == cat) for cat in cat_counts["category"]]
                sub = pl.concat(subs)
                traces.append(dict(
                    k="d", dai=dai, label=label, color=color, cat=None,
                    off=offset, n=sub.height, vis=visible,
                ))
                parts_list.append(sub)
                offset += sub.height
                continue
            for cat in cat_counts["category"]:
                sub = dsub.filter(pl.col("category") == cat)
                traces.append(dict(
                    k="d", dai=dai, label=label, color=color, cat=cat,
                    off=offset, n=sub.height, vis=visible,
                ))
                parts_list.append(sub)
                offset += sub.height
    return pl.concat(parts_list).drop("_part"), traces


def globe_params(path: Path) -> str:
    """球面座標ファイル名のタグ（md0.0_sp0.3 等）から脚注用のパラメータ表記を作る。"""
    import re
    m = re.search(r"_md([\d.]+)", path.stem)
    sp = re.search(r"_sp([\d.]+)", path.stem)
    parts = []
    if m:
        parts.append(f"min_dist={m.group(1)}")
    parts.append(f"spread={sp.group(1) if sp else '1.0'}")
    return ", " + ", ".join(parts)


def load_frame(coords_path: Path) -> tuple[pl.DataFrame, bool, bool]:
    """座標 parquet に課題の属性（種目・種別・タイトル・キーワード・大区分）を結合する。戻り値: (df, is_3d, is_globe)"""
    coords = pl.read_parquet(coords_path)
    is_3d = "c2" in coords.columns
    is_globe = "theta" in coords.columns  # 球面 UMAP（reduce --sphere）: 単位球面上の xyz
    corpus = pl.read_parquet(
        "data/processed/corpus.parquet", columns=["award_number", "kaken_id", "category"]
    )
    titles = pl.read_parquet(  # 英語タイトル補完済み
        "data/interim/awards.parquet", columns=["award_number", "title", "keywords"]
    )
    df = coords.join(corpus, on="award_number", how="left")
    df = df.join(titles, on="award_number", how="left")
    df = df.join(load_dai_labels(), on="award_number", how="left")
    df = df.with_columns(
        pl.col("kaken_id").str.extract(r"^KAKENHI-([A-Z]+)-", 1).alias("ktype")
    )
    # kaken_id が「KAKENHI-<種別>-<課題番号>」で復元できることを保証（JS側で組み立てる）
    bad = df.filter(
        pl.col("kaken_id") != "KAKENHI-" + pl.col("ktype") + "-" + pl.col("award_number")
    )
    assert bad.height == 0, f"kaken_id を分解できない行が {bad.height} 件"
    return df, is_3d, is_globe


def apply_text_colors(traces: list[dict]) -> dict | None:
    """POINT_COLOR=text のとき、大区分の色見本を所属課題の平均色に差し替える（K のように散在する区分は灰色寄りになる）。
    戻り値は凡例データ（sectors, dai）。dai モードでは None。"""
    if POINT_COLOR != "text":
        return None
    color_legend = json.loads(TEXTCOLOR_LEGEND.read_text(encoding="utf-8"))
    for t in traces:
        if t["dai"] in color_legend["dai"]:
            r, g, b = color_legend["dai"][t["dai"]]["rgb"]
            t["color"] = f"rgb({r},{g},{b})"
    return color_legend


def jitter_globe(big: pl.DataFrame, dims: list[str]) -> pl.DataFrame:
    """球面: 半径に微小な乱数を与える（不透明描画では奥行きで勝者が決まり色の偏りを防ぐ。
    半透明描画では効かないため、build_order の交互描画で偏りを平均化する。再現性のため seed 固定）"""
    rng = np.random.default_rng(42)
    r = 1.0 + np.clip(rng.normal(0.0, GLOBE_JITTER_SIGMA, big.height), -GLOBE_JITTER_CLIP, GLOBE_JITTER_CLIP)
    return big.with_columns([(pl.col(c) * pl.Series(r)).alias(c) for c in dims])


def quantize(big: pl.DataFrame, dims: list[str]) -> tuple[list[dict], list[np.ndarray]]:
    """座標の量子化: 各軸を int16 全域に線形写像（分解能=値域/65535、1ピクセル未満）。戻り値: (各軸の lo/hi, int16 配列)"""
    quant, qarrs = [], []
    for c in dims:
        v = big[c].to_numpy()
        lo, hi = float(v.min()), float(v.max())
        qarrs.append(np.round((v - lo) / (hi - lo) * 65535 - 32768).astype("<i2"))
        quant.append(dict(lo=lo, hi=hi))
    return quant, qarrs


def build_points_bin(big: pl.DataFrame, qarrs: list[np.ndarray]) -> bytes:
    """points.bin = 量子化座標（軸ごとに連続）＋ gid→uid 対応表（uint32）＋（text モード）点ごとの sRGB 各 1 バイト。

    詳細データ（シャード）は描画順でなく **課題番号順（uid）** で 1 セットだけ持ち、3 ビューで共有する（docs/shards/）。
    各ビューは gid（描画順）→ uid の対応表を points.bin の末尾に同梱する。同じ URL になるのでブラウザキャッシュが
    ビュー間で効き、切り替え時に再読み込みしない（2026-09-09）。"""
    uid_of = {a: i for i, a in enumerate(sorted(big["award_number"].to_list()))}
    uids = np.array([uid_of[a] for a in big["award_number"]], dtype="<u4")
    points_bin = b"".join(a.tobytes() for a in qarrs) + uids.tobytes()
    if POINT_COLOR == "text":
        tc = pl.read_parquet(TEXTCOLOR_PARQUET, columns=["award_number", "r", "g", "b"])
        tcj = big.select("award_number").join(tc, on="award_number", how="left")
        assert tcj["r"].null_count() == 0, "色表に無い課題がある"
        rgb = np.stack([tcj[c].to_numpy() for c in ("r", "g", "b")], axis=1).astype(np.uint8)
        points_bin += rgb.tobytes()
    return points_bin


def write_shards(big: pl.DataFrame, ktypes: list[str], cats: list[str], shard_dir: Path) -> tuple[list[str], str]:
    """共有シャード docs/shards/NNN.json（課題番号順、SHARD_SIZE 件ずつ）を書く。
    行 = [課題番号, 種別idx, タイトル, キーワード, 種目idx]。戻り値: (各片の先頭課題番号, 内容ハッシュの版)"""
    ktype_idx = {t: i for i, t in enumerate(ktypes)}
    cat_idx = {c: i for i, c in enumerate(cats)}
    master = big.sort("award_number")  # uid 順
    rows = list(zip(
        master["award_number"],
        (ktype_idx[t] for t in master["ktype"]),
        master["title"],
        ("、".join(kw) if kw is not None else "" for kw in master["keywords"]),
        (cat_idx[c] for c in master["category"]),
        strict=False,
    ))
    n_shards = (len(rows) + SHARD_SIZE - 1) // SHARD_SIZE
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_hash = hashlib.sha1()
    for s in range(n_shards):
        chunk = rows[s * SHARD_SIZE:(s + 1) * SHARD_SIZE]
        payload = json.dumps([list(r) for r in chunk], ensure_ascii=False, separators=(",", ":"))
        shard_hash.update(payload.encode("utf-8"))
        (shard_dir / f"{s:03d}.json").write_text(payload, encoding="utf-8")
    shard_first = [rows[s * SHARD_SIZE][0] for s in range(n_shards)]  # 課題番号→シャードの二分探索用
    return shard_first, shard_hash.hexdigest()[:10]


def render_html(manifest: dict, out_dir: Path, is_3d: bool, is_globe: bool) -> None:
    n = manifest["n"]
    html = (
        TEMPLATE
        .replace("__TITLE__", manifest["title"])
        .replace("__OG_DESC__", f"科研費の採択課題 {n:,}件（2019–2025年度）を研究概要の意味の近さで並べた"
                                + ("球面地図（地球儀）。" if is_globe else ("3D 地図。" if is_3d else "2D 地図。検索・絞り込み・なげなわ集計。")))
        .replace("__OG_PATH__", out_dir.name)
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
        .replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False,
                                            separators=(",", ":")))
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    coords_path = Path(sys.argv[1])
    df, is_3d, is_globe = load_frame(coords_path)
    ktypes = sorted(df["ktype"].unique().to_list())

    big, traces = build_order(df, parts=16 if is_globe else 1, merge_categories=is_globe)
    color_legend = apply_text_colors(traces)
    assert len(traces) + 2 <= 255, f"トレース数 {len(traces)} が Plotly 3D の判定上限(255)を超える"
    n = big.height
    dims = ["c0", "c1"] + (["c2"] if is_3d else [])
    if is_globe:
        big = jitter_globe(big, dims)
    quant, qarrs = quantize(big, dims)
    points_bin = build_points_bin(big, qarrs)

    out_dir = Path("docs/globe" if is_globe else f"docs/map{'3d' if is_3d else '2d'}")
    if len(sys.argv) > 2:  # 比較実験用に出力先を変えられる（例: reports/globe_compare/sp0.45）
        out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "points.bin").write_bytes(points_bin)

    cats = sorted(big["category"].unique().to_list())
    shard_dir = out_dir.parent / "shards"  # docs/shards（3 ビュー共有。内容はビューに依らず同一）
    shard_first, shard_ver = write_shards(big, ktypes, cats, shard_dir)
    n_shards = len(shard_first)
    # データ URL の版（?v=）。HTML はネットワーク優先・データはキャッシュ優先で配信するため、更新直後は
    # 新しい HTML と古い Service Worker の組み合わせが一度だけ起こり、古い points.bin が返ってくる
    # （2026-09-11、v1.3 公開直後にスマホの 2D で "Length out of range of buffer"）。URL に内容ハッシュを付ければ
    # 古いキャッシュには当たらない
    points_ver = hashlib.sha1(points_bin).hexdigest()[:10]

    manifest = dict(
        n=n, is3d=is_3d, globe=is_globe, shardSize=SHARD_SIZE, nShards=n_shards,
        pointsBytes=len(points_bin), pointsVer=points_ver, shardVer=shard_ver, quant=quant, ktypes=ktypes, cats=cats,
        traces=traces, catOrder=CATEGORY_ORDER,
        shardFirst=shard_first,
        sphereR=GLOBE_SPHERE_R if is_globe else None,
        textColor=POINT_COLOR == "text",
        colorLegend=dict(sectors=color_legend["sectors"]) if color_legend else None,
        footer=FOOTER + (f" | 球面埋め込み: output_metric=haversine{globe_params(coords_path)}" if is_globe else ""),
        title=f"科研費 学術地図 {'球面' if is_globe else ('3D' if is_3d else '2D')}",
        sub=f"2019–2025年度・{n:,}件",
    )
    render_html(manifest, out_dir, is_3d, is_globe)
    write_service_worker(out_dir.parent)

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"出力: {out_dir}/ 合計 {total / 1e6:.1f} MB "
          f"(points.bin {len(points_bin) / 1e6:.1f} MB, 共有シャード {n_shards} 個 → {shard_dir}/, トレース {len(traces)} 本, "
          f"index.html {(out_dir / 'index.html').stat().st_size / 1e3:.0f} KB)")


def write_service_worker(docs: Path) -> None:
    """docs/sw.js を書く。データ版（共有シャード＋各ビューの points.bin の内容ハッシュ）をキャッシュ名に埋め込む。

    データが変わると sw.js のバイト列が変わり、ブラウザが新しい Service Worker を入れて古いキャッシュを捨てる。
    キャッシュは取得時に貯める（先読みで二重に落とさない）。HTML はネットワーク優先（更新を即反映）、
    データ（shards/*.json, points.bin）と Plotly CDN・画像はキャッシュ優先。計測（GoatCounter）は素通し。
    """
    h = hashlib.sha1()
    for f in sorted(docs.glob("shards/*.json")) + sorted(docs.glob("*/points.bin")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    ver = h.hexdigest()[:12]
    (docs / "sw.js").write_text(SW_TEMPLATE.replace("__DATA_VER__", ver), encoding="utf-8")
    print(f"出力: {docs}/sw.js (data version {ver})")


# テンプレート（HTML/CSS/JS と Service Worker）は scripts/web/ の実ファイル。置換記号 __TITLE__ 等を埋めて出力する
WEB_DIR = Path(__file__).resolve().parent / "web"
SW_TEMPLATE = (WEB_DIR / "sw.template.js").read_text(encoding="utf-8")
TEMPLATE = (WEB_DIR / "index.template.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
