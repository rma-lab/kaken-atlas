"""ウェブ公開用の軽量インタラクティブ地図を生成する（GitHub Pages 向け）。

reports/figures/map_*_interactive.html（自己完結・約128MB）の外部データ版。
二段階読み込みで初期表示を数秒にする:

  フェーズ1  points.bin … 量子化座標(int16)。プログレスバー付きで取得→描画
  フェーズ2  shards/NNN.json … タイトル・キーワード等（SHARD_SIZE件/片）。
             描画後に背景先読み（完了で検索が有効化）。未取得片への
             ホバーはその片だけ即時取得して穴埋めする。

トレース構成・配色・UI（検索/種目フィルタ/なげなわ集計/ツールチップ/KAKENリンク）は
plot_map_interactive.py と同一仕様。kaken_id は「KAKENHI-<種別>-<課題番号>」に
分解できるため種別コードのみ持つ。

使い方:
    uv run python scripts/build_web_map.py data/processed/umap2d_nn15_md0.1.parquet
    uv run python scripts/build_web_map.py data/processed/umap3d_nn15_md0.1.parquet
    uv run python scripts/build_web_map.py <parquet> <出力先>   # 比較実験用（既定は docs/ 以下）
出力: docs/map2d/ または docs/map3d/（index.html + points.bin + shards/）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from kaken_atlas.kubun import DAI_COLORS, DAI_GLOSS, load_dai_labels  # noqa: E402
from plot_map_interactive import CATEGORY_ORDER, FOOTER  # noqa: E402

SHARD_SIZE = 2048  # 2の冪であること（JS側でビットシフトに使う）
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
        color = DAI_COLORS.get(dai, "#b9b8b0")
        label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
        traces.append(dict(
            k="a", dai=dai, label=label, color=color, n=dsub.height,
            rank=legend_order.index(dai) + 1, vis=dai != "区分なし",
        ))
    for part in range(parts):
        # 組ごとに大区分の順序を回転させ、「常に最後に描かれる大区分」を作らない
        order = draw_order[part % len(draw_order):] + draw_order[:part % len(draw_order)] if parts > 1 else draw_order
        for dai in order:
            dsub = df.filter((pl.col("dai") == dai) & (pl.col("_part") == part))
            if dsub.height == 0:
                continue
            color = DAI_COLORS.get(dai, "#b9b8b0")
            label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
            visible = dai != "区分なし"
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


def main() -> None:
    coords_path = Path(sys.argv[1])
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
    ktypes = sorted(df["ktype"].unique().to_list())
    # kaken_id が「KAKENHI-<種別>-<課題番号>」で復元できることを保証（JS側で組み立てる）
    bad = df.filter(
        pl.col("kaken_id") != "KAKENHI-" + pl.col("ktype") + "-" + pl.col("award_number")
    )
    assert bad.height == 0, f"kaken_id を分解できない行が {bad.height} 件"

    big, traces = build_order(df, parts=16 if is_globe else 1, merge_categories=is_globe)
    n_point_traces = sum(1 for t in traces if t["k"] == "d")
    assert len(traces) + 2 <= 255, f"トレース数 {len(traces)} が Plotly 3D の判定上限(255)を超える"
    n = big.height
    dims = ["c0", "c1"] + (["c2"] if is_3d else [])
    if is_globe:
        # 半径に微小な乱数を与える（不透明描画では奥行きで勝者が決まり色の偏りを防ぐ。
        # 半透明描画では効かないため、build_order の交互描画で偏りを平均化する。再現性のため seed 固定）
        rng = np.random.default_rng(42)
        r = 1.0 + np.clip(rng.normal(0.0, 0.004, n), -0.01, 0.01)
        big = big.with_columns([(pl.col(c) * pl.Series(r)).alias(c) for c in dims])

    # 座標の量子化: 各軸を int16 全域に線形写像（分解能=値域/65535、1ピクセル未満）
    quant = []
    qarrs = []
    for c in dims:
        v = big[c].to_numpy()
        lo, hi = float(v.min()), float(v.max())
        qarrs.append(np.round((v - lo) / (hi - lo) * 65535 - 32768).astype("<i2"))
        quant.append(dict(lo=lo, hi=hi))
    points_bin = b"".join(a.tobytes() for a in qarrs)

    out_dir = Path("docs/globe" if is_globe else f"docs/map{'3d' if is_3d else '2d'}")
    if len(sys.argv) > 2:  # 比較実験用に出力先を変えられる（例: reports/globe_compare/sp0.45）
        out_dir = Path(sys.argv[2])
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "points.bin").write_bytes(points_bin)

    ktype_idx = {t: i for i, t in enumerate(ktypes)}
    cats = sorted(big["category"].unique().to_list())
    cat_idx = {c: i for i, c in enumerate(cats)}
    rows = list(zip(  # [課題番号, 種別idx, タイトル, キーワード, 種目idx]
        big["award_number"],
        (ktype_idx[t] for t in big["ktype"]),
        big["title"],
        ("、".join(kw) if kw is not None else "" for kw in big["keywords"]),
        (cat_idx[c] for c in big["category"]),
        strict=False,
    ))
    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE
    for s in range(n_shards):
        chunk = rows[s * SHARD_SIZE:(s + 1) * SHARD_SIZE]
        (out_dir / "shards" / f"{s:03d}.json").write_text(
            json.dumps([list(r) for r in chunk], ensure_ascii=False,
                       separators=(",", ":")),
            encoding="utf-8",
        )

    manifest = dict(
        n=n, is3d=is_3d, globe=is_globe, shardSize=SHARD_SIZE, nShards=n_shards,
        pointsBytes=len(points_bin), quant=quant, ktypes=ktypes, cats=cats,
        traces=traces, catOrder=CATEGORY_ORDER,
        footer=FOOTER + (f" | 球面埋め込み: output_metric=haversine{globe_params(coords_path)}" if is_globe else ""),
        title=f"科研費 学術地図 {'球面' if is_globe else ('3D' if is_3d else '2D')}",
        sub=f"2019–2025年度・{n:,}件",
    )
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

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"出力: {out_dir}/ 合計 {total / 1e6:.1f} MB "
          f"(points.bin {len(points_bin) / 1e6:.1f} MB, シャード {n_shards} 個, トレース {len(traces)} 本, "
          f"index.html {(out_dir / 'index.html').stat().st_size / 1e3:.0f} KB)")


# ---------------------------------------------------------------------------
# HTML/JS テンプレート。UI仕様は plot_map_interactive.py の POST_SCRIPT を踏襲し、
# 点ごとの文字列参照を「グローバル添字 gid → シャード」の遅延解決に置き換えている。
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — KAKEN-ATLAS</title>
<meta name="description" content="__OG_DESC__">
<meta property="og:type" content="website">
<meta property="og:site_name" content="KAKEN-ATLAS">
<meta property="og:title" content="__TITLE__ — KAKEN-ATLAS">
<meta property="og:description" content="__OG_DESC__">
<meta property="og:url" content="https://rma-lab.github.io/kaken-atlas/__OG_PATH__/">
<meta property="og:image" content="https://rma-lab.github.io/kaken-atlas/ogp.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:locale" content="ja_JP">
<meta name="twitter:card" content="summary_large_image">
<link rel="canonical" href="https://rma-lab.github.io/kaken-atlas/__OG_PATH__/">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"WebPage","name":"__TITLE__ — KAKEN-ATLAS",
 "url":"https://rma-lab.github.io/kaken-atlas/__OG_PATH__/","description":"__OG_DESC__","inLanguage":"ja",
 "isPartOf":{"@id":"https://rma-lab.github.io/kaken-atlas/#website"},
 "about":{"@id":"https://rma-lab.github.io/kaken-atlas/#dataset"}}
</script>
<link rel="icon" type="image/png" href="../favicon.png">
<link rel="manifest" href="../manifest.webmanifest">
<link rel="apple-touch-icon" href="../apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<script src="__PLOTLY_CDN__" charset="utf-8"></script>
<script data-goatcounter="https://rma-lab.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>
<style>
  body { margin:0; background:#fcfcfb; }
  #plot { margin-top:48px; height:calc(100vh - 48px); touch-action:none; }
  #ka-q::-webkit-search-cancel-button { -webkit-appearance:none; appearance:none; display:none; }
  /* iOS Safari は 16px 未満の入力欄にフォーカスすると画面を自動拡大する（その後の操作にも残る）→ タッチ端末では 16px */
  @media (pointer: coarse) { #ka-q { font-size:16px !important; } }
  @media (max-width:640px) {
    #ka-bar { gap:8px !important; padding:0 10px !important; }
    #ka-title { display:none !important; }  /* タイトルはタブ・入口ページにある */
    #ka-bar input { min-width:100px !important; }
    #ka-filter-btn, #ka-help-btn { white-space:nowrap; font-size:12px; }
    .ka-sub { display:none !important; }
    #ka-results { width:86vw !important; }
  }
  #ka-loading { position:fixed; inset:0; z-index:2000; background:#fcfcfb;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:14px; font:14px -apple-system,sans-serif; color:#0b0b0b; }
  #ka-bar1-wrap { width:min(420px,80vw); height:8px; background:#e1e0d9;
    border-radius:4px; overflow:hidden; }
  #ka-bar1 { height:100%; width:0%; background:#1c5cab; transition:width .15s; }
  #ka-phase2 { position:fixed; bottom:14px; right:14px; z-index:999;
    background:#fcfcfb; border:1px solid #e1e0d9; border-radius:8px;
    box-shadow:0 3px 12px rgba(0,0,0,.1); padding:8px 14px;
    font:12px -apple-system,sans-serif; color:#52514e; display:none; }
  #ka-bar2-wrap { width:180px; height:5px; background:#e1e0d9; border-radius:3px;
    overflow:hidden; margin-top:5px; }
  #ka-bar2 { height:100%; width:0%; background:#1c5cab; transition:width .3s; }
</style>
</head>
<body>
<noscript><p style="margin:16px;font:14px/1.7 -apple-system,sans-serif">__TITLE__（KAKEN-ATLAS）: __OG_DESC__ この地図の表示には JavaScript が必要です。
概要は <a href="../">トップページ</a>、方法は <a href="https://github.com/rma-lab/kaken-atlas/blob/master/doc/method.md">技術ノート</a> を参照。
出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを KAKEN-ATLAS が編集・加工。</p></noscript>
<div id="ka-loading">
  <div><b style="font-size:16px">__TITLE__</b></div>
  <div id="ka-load-msg">マップ（点）を読み込み中…</div>
  <div id="ka-bar1-wrap"><div id="ka-bar1"></div></div>
</div>
<div id="ka-phase2">
  <span id="ka-p2-msg">詳細データを読み込み中…</span>
  <div id="ka-bar2-wrap"><div id="ka-bar2"></div></div>
</div>
<div id="plot"></div>
<script>
'use strict';
var M = __MANIFEST__;
var is3d = M.is3d;
var isGlobe = !!M.globe;  // 球面ビュー: 単位球面上の点＋不透明な球（裏側の点を隠して地球儀に見せる）
var SHARD_SHIFT = Math.log2(M.shardSize);
var isTouch = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
var narrow = window.innerWidth < 640;  // スマホ幅: 凡例をドロワー化・ヘッダー圧縮

// ==== デザイントークン ====
var INK = '#0b0b0b', MUTED = '#898781', SUB = '#52514e', LINE = '#e1e0d9';
var SURFACE = '#fcfcfb';
var PANEL = 'background:#fcfcfb;border:1px solid ' + LINE + ';border-radius:8px;' +
  'box-shadow:0 3px 12px rgba(0,0,0,0.10);font:12.5px/1.7 -apple-system,sans-serif;color:' + INK;

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function fmt(x) { return x.toLocaleString('ja-JP'); }

// ==== 詳細データ（シャード）: det[s] = [[課題番号, 種別idx, タイトル, キーワード], ...] ====
var det = new Array(M.nShards).fill(null);
var detPending = new Array(M.nShards).fill(null);
var detLoaded = 0, allLoaded = false;
function pad3(s) { return String(s).padStart(3, '0'); }
function ensureShard(s) {
  if (det[s]) return Promise.resolve(det[s]);
  if (detPending[s]) return detPending[s];
  detPending[s] = fetch('shards/' + pad3(s) + '.json')
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (rows) {
      det[s] = rows; detPending[s] = null; detLoaded++;
      updatePhase2();
      return rows;
    })
    .catch(function (e) { detPending[s] = null; throw e; });
  return detPending[s];
}
function getRow(gid) {
  var d = det[gid >> SHARD_SHIFT];
  return d ? d[gid & (M.shardSize - 1)] : null;
}
function kakenId(row) { return 'KAKENHI-' + M.ktypes[row[1]] + '-' + row[0]; }
// 種目名: 詳細行があれば行から（球面は大区分×組でトレースを束ねるため）、なければトレースの種目
function catOf(gid, row) {
  if (row && row.length > 4 && M.cats) return M.cats[row[4]];
  return M.traces[traceOf[gid]].cat || '';
}

// ==== フェーズ1: points.bin をプログレス付きで取得 ====
var bar1 = document.getElementById('ka-bar1');
var loadMsg = document.getElementById('ka-load-msg');
async function loadPoints() {
  var r = await fetch('points.bin');
  if (!r.ok) throw new Error('points.bin: ' + r.status);
  var reader = r.body.getReader();
  var chunks = [], recv = 0;
  for (;;) {
    var c = await reader.read();
    if (c.done) break;
    chunks.push(c.value); recv += c.value.length;
    bar1.style.width = Math.min(100, recv / M.pointsBytes * 100) + '%';
  }
  var buf = new Uint8Array(recv), o = 0;
  for (var i = 0; i < chunks.length; i++) { buf.set(chunks[i], o); o += chunks[i].length; }
  return buf.buffer;
}

// ==== 起動 ====
var plot = document.getElementById('plot');
var xs, ys, zs, traceOf, gidOffset;
main().catch(function (e) {
  loadMsg.innerHTML = '読み込みに失敗しました（' + esc(e.message) + '）。<br>' +
    'HTTPサーバ経由で開いているか、通信状態を確認してください。';
});

async function main() {
  var buf = await loadPoints();
  loadMsg.textContent = 'マップを描画中…';
  bar1.style.width = '100%';
  await new Promise(function (res) { setTimeout(res, 30); });  // 描画前にUI更新を反映

  var N = M.n, ndim = is3d ? 3 : 2;
  var i16 = new Int16Array(buf);
  function deq(k) {
    var q = M.quant[k], out = new Float32Array(N), base = k * N;
    var scale = (q.hi - q.lo) / 65535;
    for (var i = 0; i < N; i++) out[i] = (i16[base + i] + 32768) * scale + q.lo;
    return out;
  }
  xs = deq(0); ys = deq(1); if (is3d) zs = deq(2);

  // gid → トレース番号（検索の表示判定用）と、トレース番号 → 先頭gid
  traceOf = new Uint16Array(N);
  gidOffset = [];
  var data = [];
  M.traces.forEach(function (t, ti) {
    if (t.k === 'a') {  // 凡例アンカー（空トレース）: 凡例は常在のアンカーが担う
      var a = {
        mode: 'markers', x: [null], y: [null],
        name: t.label + ' ' + fmt(t.n), legendgroup: t.dai, legendrank: t.rank,
        showlegend: true, hoverinfo: 'none',
        visible: t.vis ? true : 'legendonly',
        marker: { size: 6, color: t.color, opacity: 0.9 },
        type: is3d ? 'scatter3d' : 'scattergl',
      };
      if (is3d) a.z = [null];
      data.push(a); gidOffset.push(-1);
      return;
    }
    var end = t.off + t.n;
    for (var g = t.off; g < end; g++) traceOf[g] = ti;
    var d = {
      mode: 'markers',
      x: xs.subarray(t.off, end), y: ys.subarray(t.off, end),
      name: t.label + ' ' + fmt(t.n), meta: t.cat, legendgroup: t.dai,
      showlegend: false, hoverinfo: 'none',
      visible: t.vis ? true : 'legendonly',
      marker: is3d ? { size: isGlobe ? 1.4 : 1.3, color: t.color, opacity: isGlobe ? 0.75 : 0.55 }
                   : { size: 2.2, color: t.color, opacity: 0.5 },
      type: is3d ? 'scatter3d' : 'scattergl',
    };
    if (is3d) d.z = zs.subarray(t.off, end);
    data.push(d); gidOffset.push(t.off);
  });
  if (isGlobe) {  // 半径 0.985 の不透明な球を末尾に追加（data 添字＝M.traces 添字を保つ）。gidOffset は -1（点ではない）
    var NU = 60, NV = 30, gx = [], gy = [], gz = [];
    for (var iv = 0; iv <= NV; iv++) {
      var th = Math.PI * iv / NV, rx = [], ry = [], rz = [];
      for (var iu = 0; iu <= NU; iu++) {
        var ph = 2 * Math.PI * iu / NU;
        rx.push(0.985 * Math.sin(th) * Math.cos(ph)); ry.push(0.985 * Math.sin(th) * Math.sin(ph)); rz.push(0.985 * Math.cos(th));
      }
      gx.push(rx); gy.push(ry); gz.push(rz);
    }
    data.push({ type: 'surface', x: gx, y: gy, z: gz, showscale: false, hoverinfo: 'none', showlegend: false,
      colorscale: [[0, '#f3f2ec'], [1, '#f3f2ec']], opacity: 1,
      lighting: { ambient: 0.9, diffuse: 0.3, specular: 0.02, roughness: 0.9 }, contours: { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } } });
    gidOffset.push(-1);
  }

  var layout = {
    paper_bgcolor: SURFACE,
    legend: { itemsizing: 'constant', font: { size: 11 }, groupclick: 'togglegroup' },
    annotations: [{ text: M.footer, x: 0, y: 0, xref: 'paper', yref: 'paper',
                    showarrow: false, font: { size: 9, color: MUTED } }],
    margin: { l: 0, r: 0, t: 24, b: 30 },
  };
  if (is3d) {
    layout.scene = {
      xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false },
      aspectmode: 'data', bgcolor: SURFACE, dragmode: 'orbit',
    };
    if (isGlobe) {  // 球全体が収まる距離から見る。回転は自由（軸は固定しない）
      layout.scene.camera = { eye: { x: 1.9, y: 1.4, z: 0.9 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
      layout.scene.aspectmode = 'cube';
    }
  } else {
    layout.plot_bgcolor = SURFACE;
    layout.xaxis = { visible: false };
    layout.yaxis = { visible: false, scaleanchor: 'x' };
    layout.dragmode = 'pan';
  }
  await Plotly.newPlot(plot, data, layout,
    { scrollZoom: true, displaylogo: false, doubleClick: 'reset', responsive: true,
      modeBarButtonsToRemove: isGlobe ? ['pan3d'] : [] });
  if (isGlobe) {  // 注視点が動いたら（右ドラッグの移動など）中心へ戻す。視点も同じだけ戻して見え方を保つ
    plot.on('plotly_relayout', function (ev) {
      var cam = plot._fullLayout.scene && plot._fullLayout.scene.camera;
      if (!cam || !cam.center) return;
      var c = cam.center;
      if (Math.abs(c.x) < 1e-6 && Math.abs(c.y) < 1e-6 && Math.abs(c.z) < 1e-6) return;
      Plotly.relayout(plot, { 'scene.camera.center': { x: 0, y: 0, z: 0 },
        'scene.camera.eye': { x: cam.eye.x - c.x, y: cam.eye.y - c.y, z: cam.eye.z - c.z } });
    });
  }

  // 狭い画面（スマホ）: 凡例は常時表示せずドロワー（setupUI で生成）に委ね、地図を全面に
  if (narrow) {
    var mobile = { showlegend: false, 'annotations[0].visible': false };
    if (!is3d) mobile['margin.b'] = 40;
    await Plotly.relayout(plot, mobile);
  }

  document.getElementById('ka-loading').style.display = 'none';
  setupUI();
  startPrefetch();
}

// ==== フェーズ2: 背景先読み（同時4本）。完了で検索が有効になる ====
function updatePhase2() {
  document.getElementById('ka-bar2').style.width = (detLoaded / M.nShards * 100) + '%';
  document.getElementById('ka-p2-msg').textContent =
    '詳細データを読み込み中… ' + detLoaded + '/' + M.nShards;
}
function startPrefetch() {
  var p2 = document.getElementById('ka-phase2');
  p2.style.display = 'block';
  updatePhase2();
  var next = 0;
  function pump() {
    while (next < M.nShards && (det[next] || detPending[next])) next++;
    if (next >= M.nShards) {
      if (detLoaded >= M.nShards) finishPrefetch();
      return;
    }
    ensureShard(next++).catch(function () {}).then(function () { pump(); });
  }
  for (var i = 0; i < 4; i++) pump();
}
function finishPrefetch() {
  if (allLoaded) return;
  allLoaded = true;
  var q = document.getElementById('ka-q');
  if (q) { q.disabled = false; q.placeholder = 'タイトル・キーワード・課題番号を検索'; }
  var p2 = document.getElementById('ka-phase2');
  document.getElementById('ka-bar2-wrap').style.display = 'none';
  document.getElementById('ka-p2-msg').textContent = '✓ 全データ読み込み完了（検索が使えます）';
  setTimeout(function () { p2.style.display = 'none'; }, 4000);
}

// ==== UI一式（plot_map_interactive.py の POST_SCRIPT を移植） ====
function setupUI() {

// ---- ヘッダーバー ----
var bar = document.createElement('div');
bar.id = 'ka-bar';
bar.style.cssText = 'position:fixed;top:0;left:0;right:0;height:48px;z-index:998;' +
  'display:flex;align-items:center;gap:14px;padding:0 18px;background:#fcfcfb;' +
  'border-bottom:1px solid ' + LINE + ';font:13px -apple-system,sans-serif;color:' + INK;
// 左端＝サイトメニュー（羅針盤アイコン。PC はタイトルも含めてホバー、スマホはアイコンのみタップ）:
// トップページへ戻る＋他ビューへの直接移動。ウェブアプリ（standalone）モードには戻るボタンがないため必須
var curView = isGlobe ? 'globe' : (is3d ? 'map3d' : 'map2d');
var siteMenu = [['../', 'トップページ', ''], ['../map2d/', '2D 地図', 'map2d'],
                ['../map3d/', '3D 地図', 'map3d'], ['../globe/', '球面地図', 'globe']].map(function (v, i) {
  var item = v[2] === curView
    ? '<div style="padding:6px 16px;color:' + MUTED + ';white-space:nowrap">' + v[1] + '<span style="font-size:11px">（表示中）</span></div>'
    : '<a href="' + v[0] + '" style="display:block;padding:6px 16px;color:' + INK + ';text-decoration:none;white-space:nowrap">' + v[1] + '</a>';
  return item + (i === 0 ? '<div style="border-top:1px solid ' + LINE + ';margin:4px 0"></div>' : '');
}).join('');
bar.innerHTML =
  '<div id="ka-home-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span id="ka-home-btn" style="display:flex;align-items:center;gap:8px;cursor:default;white-space:nowrap">' +
  '    <img src="../icon-192.png" width="24" height="24" alt="メニュー" style="display:block;border-radius:6px">' +
  '    <span id="ka-title"><b style="font-size:14.5px">' + M.title + '</b>' +
  '     <span class="ka-sub" style="color:' + MUTED + ';font-size:12px">' + M.sub + '</span>' +
  '     <span style="color:' + SUB + ';font-size:11px"> ▾</span></span>' +
  '  </span>' +
  '  <div id="ka-home-body" style="display:none;position:absolute;top:100%;left:0;min-width:160px;padding:6px 0;' + PANEL + '">' +
  siteMenu + '</div>' +
  '</div>' +
  '<div id="ka-q-wrap" style="position:relative;flex:0 1 300px;min-width:170px">' +
  '  <input id="ka-q" type="search" placeholder="詳細データ読み込み中…" disabled' +
  '   style="width:100%;box-sizing:border-box;padding:6px 30px 6px 12px;border:1px solid #cfcec7;' +
  '   border-radius:15px;background:#fff;font:12.5px -apple-system,sans-serif;outline:none">' +
  '  <span id="ka-q-clear" title="検索をクリア" style="display:none;position:absolute;right:6px;top:50%;' +
  '   transform:translateY(-50%);width:20px;height:20px;line-height:20px;text-align:center;border-radius:10px;' +
  '   background:#d8d7d0;color:#fff;font-size:14px;cursor:pointer;user-select:none">×</span>' +
  '  <div id="ka-results" style="display:none;position:absolute;top:36px;left:0;width:380px;' +
  '   max-height:55vh;overflow-y:auto;padding:8px 12px;' + PANEL + '"></div>' +
  '</div>' +
  '<div style="flex:1"></div>' +
  '<div id="ka-dai-wrap" style="display:none;align-self:stretch;align-items:center">' +
  '  <span id="ka-dai-btn" style="cursor:default;color:' + SUB + ';white-space:nowrap;font-size:12px">大区分 ▾</span>' +
  '</div>' +
  '<div id="ka-filter-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span id="ka-filter-btn" style="cursor:default;color:' + SUB + '">' + (narrow ? '種目 ▾' : '種目フィルタ ▾') + '</span>' +
  '  <div id="ka-body" style="display:none;position:absolute;top:100%;right:0;' +
  '   max-height:70vh;overflow-y:auto;padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>' +
  '<div id="ka-help-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span id="ka-help-btn" style="cursor:default;color:' + SUB + '">操作 ▾</span>' +
  '  <div id="ka-help-body" style="display:none;position:absolute;top:100%;right:0;' +
  '   padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>';
document.body.insertBefore(bar, document.body.firstChild);
if (allLoaded) finishPrefetch();  // 先読みがヘッダー生成より先に終わった場合の取りこぼし

// メニュー開閉。マウス=ホバー（斜め移動で一瞬外に出ても閉じない350ms猶予）、
// タッチ=ボタンタップでトグル・外側タップで閉じる
var menus = [['ka-home-wrap', 'ka-home-body', 'ka-home-btn'],
             ['ka-filter-wrap', 'ka-body', 'ka-filter-btn'],
             ['ka-help-wrap', 'ka-help-body', 'ka-help-btn']];
menus.forEach(function (m) {
  var wrap = document.getElementById(m[0]);
  var body = document.getElementById(m[1]);
  if (isTouch) {
    document.getElementById(m[2]).addEventListener('click', function () {
      body.style.display = body.style.display === 'none' ? 'block' : 'none';
    });
    return;
  }
  var closeTimer = null;
  wrap.addEventListener('mouseenter', function () {
    if (closeTimer) { clearTimeout(closeTimer); closeTimer = null; }
    body.style.display = 'block';
  });
  wrap.addEventListener('mouseleave', function () {
    closeTimer = setTimeout(function () { body.style.display = 'none'; }, 350);
  });
});
if (isTouch) {
  document.addEventListener('click', function (e) {
    menus.forEach(function (m) {
      var wrap = document.getElementById(m[0]);
      if (!wrap.contains(e.target)) document.getElementById(m[1]).style.display = 'none';
    });
  });
}

if (isGlobe) {  // 球面はトレースを大区分×組で束ねるため種目単位の表示切替ができない
  document.getElementById('ka-filter-wrap').style.display = 'none';
}
document.getElementById('ka-help-body').innerHTML = isTouch
  ? (is3d ? '<div>1本指: 回転 / 2本指ピンチ: 拡大縮小' + (isGlobe ? '' : ' / 2本指ドラッグ: 移動') + '</div>'
          : '<div>1本指: 移動 / 2本指ピンチ: 拡大縮小</div>' +
            '<div>ダブルタップ: 全体表示に戻る</div>') +
    '<div>点をタップ: 詳細カード / カードをタップ: KAKENページ</div>' +
    '<div>「大区分」: 表示切替・「のみ」でその区分だけ・すべて表示/非表示</div>'
  : is3d
  ? '<div>ドラッグ: 回転 / スクロール: 拡大縮小</div>' +
    '<div>点にホバー: 概要 / クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替 / ダブルクリック: その大区分だけ表示</div>'
  : '<div>スクロール: 拡大縮小 / ドラッグ: 移動</div>' +
    '<div>ダブルクリック: 全体表示に戻る</div>' +
    '<div>点にホバー: 概要 / クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替 / ダブルクリック: その大区分だけ表示</div>' +
    '<div>ツールバーのなげなわ/矩形: 囲って集計</div>' +
    '<div>Esc: 選択解除</div>';

// ---- 点の詳細表示 ----
// PC（マウス）: ホバーでプレビュー（カーソル追従・操作不可）、クリックで即 KAKEN ページを開く。
//   3つのビュー（2D/3D/球面）で共通。カード・リングは使わない（2026-09-05 ユーザ仕様）
// タッチ端末: タップで詳細カード。カードは点と重ならない位置に置き、点はリングで強調する。
//   カードのどこをタップしても KAKEN ページが開く（本物のリンク。スクリプトからの
//   新規タブ起動は iOS で弾かれることがあるため）。× または点のない場所のタップで閉じる。
var mx = 0, my = 0, hoveredGid = null, suppressUntil = 0;
document.addEventListener('mousemove', function (e) {
  mx = e.clientX; my = e.clientY;
  if (tip.style.display !== 'none') placeTip();
});
// PC: 地図を動かし始めたら（ドラッグ／ホイール）プレビューを消し、操作が終わるまで再表示しない
// （2026-09-07 ユーザ仕様。3D は回転中に hover が飛ばず古い吹き出しが残っていた）
var tipMuteUntil = 0, dragFrom = null;
if (!isTouch) {
  plot.addEventListener('mousedown', function (e) { dragFrom = [e.clientX, e.clientY]; }, true);
  document.addEventListener('mousemove', function (e) {
    if (!dragFrom) return;
    if (Math.abs(e.clientX - dragFrom[0]) > 3 || Math.abs(e.clientY - dragFrom[1]) > 3) {
      tipMuteUntil = Infinity; hoverNone();
    }
  });
  document.addEventListener('mouseup', function () {
    if (tipMuteUntil === Infinity) tipMuteUntil = Date.now() + 250;
    dragFrom = null;
  });
  plot.addEventListener('wheel', function () { tipMuteUntil = Date.now() + 350; hoverNone(); },
                        { capture: true, passive: true });
}
function gidOf(p) {
  if (!p || !(gidOffset[p.curveNumber] >= 0)) return null;  // 強調リング等の補助トレースは対象外
  return gidOffset[p.curveNumber] + p.pointNumber;
}
function kakenUrl(row) { return 'https://kaken.nii.ac.jp/ja/grant/' + kakenId(row) + '/'; }
var ELL = 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
function headerHtml(tr, closable) {
  return '<div style="' + ELL + ';background:' + tr.color + ';color:#fff;font-weight:600;' +
    'margin:-6px -9px 4px -9px;padding:4px 9px;border-radius:4.5px 4.5px 0 0;position:relative">' +
    esc(tr.label) +
    (closable ? '<span data-close="1" style="position:absolute;right:0;top:0;padding:4px 12px;' +
                'font-size:15px;line-height:1.4;cursor:pointer">×</span>' : '') + '</div>';
}

// ホバー用プレビュー（マウスのみ）
var tip = document.createElement('div');
tip.id = 'ka-tip';
tip.style.cssText = 'position:fixed;display:none;z-index:1000;background:#fff;' +
  'border:1.5px solid #999;border-radius:6px;padding:6px 9px;pointer-events:none;width:320px;' +
  'font:12px/1.5 -apple-system,sans-serif;color:' + INK + ';box-shadow:0 2px 8px rgba(0,0,0,0.15)';
document.body.appendChild(tip);
function placeTip() {  // 画面外にはみ出さないようクランプ
  tip.style.left = Math.max(4, Math.min(mx + 16, window.innerWidth - 336)) + 'px';
  tip.style.top = Math.max(52, Math.min(my + 12, window.innerHeight - 100)) + 'px';
}
function renderTip(gid, tr) {
  if (Date.now() < tipMuteUntil) return;  // 地図の操作中は出さない
  var row = getRow(gid);
  var title = row ? esc((row[2] || '（タイトルなし）').slice(0, 48))
                  : '<span style="color:' + MUTED + '">（読み込み中…）</span>';
  var tail = row ? esc(catOf(gid, row) + ' / ' + row[0]) : esc(catOf(gid, null));
  tip.innerHTML = headerHtml(tr, false) +
    '<div style="' + ELL + '">' + title + '</div>' +
    '<div style="' + ELL + '">' + tail + '</div>' +
    '<div style="color:' + MUTED + ';font-size:11px">クリックでKAKENページを開く</div>';
  tip.style.borderColor = tr.color;
  tip.style.display = 'block'; placeTip();
}

// 3D の補完判定: Plotly の 3D は画素ぴったりの判定しかせず、小さい半透明の点では「点の上」でも
// 下地や空白と判定されることが多い。カメラ行列で全点を画面に投影し、カーソル最寄りの点（許容 tol px、
// 同点なら手前）を返す。20万点の投影は数ミリ秒
function proj3dSetup() {  // 現在のカメラの「データ座標→画面座標」行列と、球面の可視判定用の視点
  var sc = plot._fullLayout.scene && plot._fullLayout.scene._scene;
  if (!sc || !sc.glplot || !sc.glplot.cameraParams) return null;
  var cp = sc.glplot.cameraParams, ds = sc.dataScale || [1, 1, 1];
  var canvas = plot.querySelector('canvas'), rect = canvas.getBoundingClientRect();
  function mul(a, b) {  // 4x4 列優先 a*b
    var o = new Float64Array(16);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
      var v = 0; for (var k = 0; k < 4; k++) v += a[k * 4 + i] * b[j * 4 + k];
      o[j * 4 + i] = v;
    }
    return o;
  }
  var m = mul(cp.projection, mul(cp.view, cp.model));
  var E = null;
  if (isGlobe) {
    var v = cp.view;
    E = [-(v[0] * v[12] + v[1] * v[13] + v[2] * v[14]),
         -(v[4] * v[12] + v[5] * v[13] + v[6] * v[14]),
         -(v[8] * v[12] + v[9] * v[13] + v[10] * v[14])];
  }
  return { m: m, ds: ds, E: E, hw: rect.width / 2, hh: rect.height / 2, ox: rect.left + rect.width / 2, oy: rect.top + rect.height / 2 };
}
plot._project3d = function (g) { return project3d(g); };  // 検証用
function project3d(g, P) {  // 1点の画面座標 [x, y]（裏側・カメラ後方なら null）
  P = P || proj3dSetup(); if (!P) return null;
  var m = P.m, x = xs[g] * P.ds[0], y = ys[g] * P.ds[1], z = zs[g] * P.ds[2];
  if (P.E && x * P.E[0] + y * P.E[1] + z * P.E[2] <= x * x + y * y + z * z) return null;
  var w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w <= 0) return null;
  return [P.ox + (m[0] * x + m[4] * y + m[8] * z + m[12]) / w * P.hw,
          P.oy - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * P.hh];
}
function nearestGid3d(cx, cy, tol) {
  var P = proj3dSetup(); if (!P) return null;
  var m = P.m, ds = P.ds, E = P.E;
  var vis = M.traces.map(function (t, ti) { return t.k === 'd' && traceVisible(ti); });
  // 球面: 球に隠れる裏側の点を除外する（接平面条件 dot(P, E) > dot(P, P)。E は proj3dSetup で計算）
  var hw = P.hw, hh = P.hh, ox = P.ox, oy = P.oy;
  var best = null, bd = tol * tol, bz = Infinity;
  for (var g = 0; g < M.n; g++) {
    if (!vis[traceOf[g]]) continue;
    var x = xs[g] * ds[0], y = ys[g] * ds[1], z = zs[g] * ds[2];
    if (E && x * E[0] + y * E[1] + z * E[2] <= x * x + y * y + z * z) continue;  // 球の裏側
    var w = m[3] * x + m[7] * y + m[11] * z + m[15];
    if (w <= 0) continue;
    var sx = ox + (m[0] * x + m[4] * y + m[8] * z + m[12]) / w * hw;
    var sy = oy - (m[1] * x + m[5] * y + m[9] * z + m[13]) / w * hh;
    var dx = sx - cx, dy = sy - cy, d2 = dx * dx + dy * dy;
    if (d2 > bd) continue;
    var nz = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;  // 小さいほど手前
    if (best === null || nz < bz - 1e-4 || (Math.abs(nz - bz) <= 1e-4 && d2 < bd)) { best = g; bd = Math.max(d2, 0); bz = nz; }
  }
  return best;
}
function hoverGid(gid) {  // 点にホバーした（Plotly 判定 or 補完判定）
  hoveredGid = gid;
  if (resolveTap(gid)) return;
  if (isTouch || selGid !== null) return;
  var tr = M.traces[traceOf[gid]];
  renderTip(gid, tr);
  if (!getRow(gid)) {
    ensureShard(gid >> SHARD_SHIFT).then(function () {
      if (hoveredGid === gid && selGid === null) renderTip(gid, tr);
    }).catch(function () {});
  }
}
function hoverNone() { hoveredGid = null; tip.style.display = 'none'; }
// 補完判定に使う位置: 待ち中のタップがあればその位置、タッチ端末は直前のタップ位置、PC はカーソル位置
// （スマホは合成マウスイベントが来ない環境があり、mx/my が古いままのことがある）
function pointerXY() {
  if (pendingTap) return [pendingTap.x, pendingTap.y];
  return [mx, my];
}
plot.on('plotly_hover', function (d) {
  if (isTouch) return;  // タッチはタップの自前判定のみ
  var gid = gidOf(d.points[0]);
  if (gid === null) {  // 球面の下地など点以外: 3D は補完判定で最寄りの点を探し、なければ外れた扱い
    var xy = is3d ? pointerXY() : null;
    var g = xy ? nearestGid3d(xy[0], xy[1], isTouch ? 14 : 8) : null;
    if (g !== null) hoverGid(g); else hoverNone();
    return;
  }
  hoveredGid = gid;
  if (resolveTap(gid)) return;      // タップ/クリック直後の遅れて届いた判定 → その点を選択
  if (isTouch || selGid !== null) return;  // タッチ端末はプレビューなし。カード表示中もプレビューは出さない
  var tr = M.traces[traceOf[gid]];
  renderTip(gid, tr);
  if (!getRow(gid)) {  // 未取得シャードはその場で取得し、まだ同じ点なら描き直す
    ensureShard(gid >> SHARD_SHIFT).then(function () {
      if (hoveredGid === gid && selGid === null) renderTip(gid, tr);
    }).catch(function () {});
  }
});
plot.on('plotly_unhover', function () {
  if (isTouch) return;
  // 3D の空白判定も補完（小さい点の隙間にカーソルが落ちたとき）。カーソルがプロット外なら外れ
  if (is3d) {
    var xy = pointerXY(), r = plot.getBoundingClientRect();
    if (xy && xy[0] >= r.left && xy[0] <= r.right && xy[1] >= r.top && xy[1] <= r.bottom) {
      var g = nearestGid3d(xy[0], xy[1], isTouch ? 14 : 8);
      if (g !== null) { hoverGid(g); return; }
    }
  }
  hoverNone();
});

// 詳細カード（選択中の点）
var card = document.createElement('div');
card.id = 'ka-card';
card.style.cssText = 'position:fixed;display:none;z-index:1001;background:#fff;' +
  'border:2px solid #999;border-radius:8px;padding:6px 9px;width:320px;box-sizing:border-box;' +
  'font:12.5px/1.5 -apple-system,sans-serif;color:' + INK + ';box-shadow:0 4px 16px rgba(0,0,0,0.22)';
if (narrow) card.style.cssText += ';left:8px;right:8px;width:auto;font-size:13px';
document.body.appendChild(card);
var selGid = null, selXY = null;

// 強調リング: DOM 要素を点の画面位置に重ねる（Plotly のトレース更新は 20万点の再描画を伴い、
// gl3d では極端に遅くなるため使わない）。2Dはパン・ズーム後に再投影して追随、3Dは回転で消す
var ring = document.createElement('div');
ring.id = 'ka-ring';
var RING = 16;  // 直径(px)
ring.style.cssText = 'position:fixed;display:none;z-index:997;width:' + RING + 'px;height:' + RING + 'px;' +
  'border-radius:50%;box-sizing:border-box;border:2.5px solid ' + INK + ';' +
  'box-shadow:0 0 0 1.5px #fff,inset 0 0 0 1.5px #fff;pointer-events:none';
document.body.appendChild(ring);
function showRing(x, y) {
  ring.style.left = (x - RING / 2) + 'px'; ring.style.top = (y - RING / 2) + 'px'; ring.style.display = 'block';
}
function hideRing() { ring.style.display = 'none'; }

function renderCard(gid, tr) {
  var row = getRow(gid);
  // タイトルは2行分の高さで固定（1行でもカードの高さが変わらない。3行以上は省略記号）
  var body =
    '<div style="font-weight:600;line-height:1.4;min-height:2.8em;margin:2px 0;overflow:hidden;' +
    'display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">' +
    (row ? esc(row[2] || '（タイトルなし）') : '<span style="color:' + MUTED + '">（読み込み中…）</span>') +
    '</div>' +
    '<div style="' + ELL + ';color:' + SUB + '">' + esc(row ? catOf(gid, row) + ' / ' + row[0] : catOf(gid, null)) + '</div>' +
    '<div style="' + ELL + ';color:' + MUTED + ';font-size:11.5px;min-height:1.5em">' + esc(row ? row[3] : '') + '</div>';
  var inner = headerHtml(tr, true) + body;
  card.innerHTML = row
    ? '<a data-open="1" href="' + esc(kakenUrl(row)) + '" target="_blank" rel="noopener"' +
      ' style="display:block;color:inherit;text-decoration:none;cursor:pointer">' + inner + '</a>'
    : inner;
  card.style.borderColor = tr.color;
  card.style.display = 'block';
}
function placeCard(cx, cy) {  // 点（画面座標）と重ならない位置に置く
  var W = window.innerWidth, H = window.innerHeight;
  if (narrow) {  // スマホ幅: 基本は下端（大区分ボタンの上）。点が下端のカードと重なる位置のときだけ上端
    card.style.top = ''; card.style.bottom = '';
    var bottomTop = H - 64 - 34 - card.offsetHeight - 24;  // 下端カードの上辺（safe-area 最大34px と余白を見込む）
    if (cy > bottomTop) card.style.top = '56px';
    else card.style.bottom = 'calc(64px + env(safe-area-inset-bottom))';
    return;
  }
  var w = card.offsetWidth, h = card.offsetHeight, gap = 28;
  var left = cx + gap;                       // 原則は点の右側
  if (left + w > W - 8) left = cx - gap - w;  // 入らなければ左側
  if (left < 8) left = 8;
  card.style.left = left + 'px';
  card.style.top = Math.max(56, Math.min(cy - h / 2, H - h - 8)) + 'px';
}
function selectPoint(gid, cx, cy) {
  if (!is3d) { var pj = projected(gid); cx = pj[0]; cy = pj[1]; }  // 2Dは点の正確な位置に吸着
  selGid = gid; selXY = [cx, cy];
  var tr = M.traces[traceOf[gid]];
  tip.style.display = 'none';
  showRing(cx, cy);
  renderCard(gid, tr); placeCard(cx, cy);
  if (!getRow(gid)) {
    ensureShard(gid >> SHARD_SHIFT).then(function () {
      if (selGid === gid) { renderCard(gid, tr); placeCard(selXY[0], selXY[1]); }
    }).catch(function () {});
  }
}
function clearSelection() {
  if (selGid === null) return;
  selGid = null; selXY = null; card.style.display = 'none'; hideRing();
}
// PC: クリックで即 KAKEN ページ（新規タブ）。DOM click と plotly_click の両方から呼ばれ得るので 600ms で重複抑止
var lastK = null, lastT = 0;
function openKaken(gid) {
  var row = getRow(gid);
  if (!row) {  // 詳細未取得: 取得後、ユーザ操作の有効期間内（transient activation）なら開く。期限切れなら次のクリックで
    ensureShard(gid >> SHARD_SHIFT).then(function () {
      var ua = navigator.userActivation;
      if (ua && ua.isActive) openKaken(gid);
    }).catch(function () {});
    return;
  }
  var k = kakenId(row), now = Date.now();
  if (k === lastK && now - lastT < 600) return;
  lastK = k; lastT = now;
  var a = document.createElement('a');
  a.href = kakenUrl(row); a.target = '_blank'; a.rel = 'noopener';
  document.body.appendChild(a); a.click(); a.remove();
}
// 点が確定したときの動作: PC は即オープン、タッチはカード
function actOn(gid, cx, cy) { if (isTouch) selectPoint(gid, cx, cy); else openKaken(gid); }
card.addEventListener('click', function (e) {
  if (e.target.getAttribute('data-close')) { e.preventDefault(); clearSelection(); }
  // それ以外はアンカーの既定動作（新規タブで KAKEN ページ）
});
function projected(gid) {  // 2D: データ座標→画面座標
  var fl = plot._fullLayout, rect = plot.getBoundingClientRect();
  var xr = fl.xaxis.range, yr = fl.yaxis.range;
  return [rect.left + fl._size.l + (xs[gid] - xr[0]) / (xr[1] - xr[0]) * fl._size.w,
          rect.top + fl._size.t + (yr[1] - ys[gid]) / (yr[1] - yr[0]) * fl._size.h];
}

// ---- クリック/タップ → 選択 ----
plot.on('plotly_doubleclick', function () { suppressUntil = Date.now() + 700; });
// 2D: パン・ピンチ直後のクリック/タップは無視（relayout を合図に抑止）。選択中はカードを点に追随
plot.on('plotly_relayout', function () {
  if (is3d) return;
  suppressUntil = Date.now() + 400;
  if (selGid !== null) { selXY = projected(selGid); showRing(selXY[0], selXY[1]); placeCard(selXY[0], selXY[1]); }
});
// 3D: gl3d はただのクリック/タップでも relayout を出すため relayout は使えない。
// 押下→離す間に実際に動かした（回転した）操作かどうかを記録し、その操作由来のクリックは無視する
var lastTouchXY = null, lastTouchAt = 0;  // 3Dタッチ: plotly_click に座標が乗らないため直前のタッチ位置を使う
var gestureMoved = false;                 // 直近の押下→離す操作で動いたか（3Dのみ更新）
if (is3d) {
  var down3 = null;
  function down3End(x, y) {
    gestureMoved = !!down3 && (Math.abs(x - down3[0]) > 10 || Math.abs(y - down3[1]) > 10);
    if (gestureMoved) clearSelection();  // 回転したら選択（リング・カード）は閉じる
    down3 = null;
  }
  plot.addEventListener('mousedown', function (e) { down3 = [e.clientX, e.clientY]; }, true);
  plot.addEventListener('mouseup', function (e) { down3End(e.clientX, e.clientY); }, true);
  var down3At = 0;
  plot.addEventListener('touchstart', function (e) {
    down3 = (e.touches.length === 1) ? [e.touches[0].clientX, e.touches[0].clientY] : null;
    down3At = Date.now();
  }, { capture: true, passive: true });
  plot.addEventListener('touchend', function (e) {
    if (e.touches.length || !e.changedTouches.length) return;
    var c = e.changedTouches[0];
    var wasTap = !!down3 && Date.now() - down3At < 500;
    down3End(c.clientX, c.clientY);
    lastTouchXY = [c.clientX, c.clientY]; lastTouchAt = Date.now();
    // タッチ端末の 3D/球面: タップは Plotly の判定を使わず、タップ位置から自前で最寄りの点を決める
    // （合成マウスイベントや判定バッファ、描画フレーム待ちに依存しないので実機で安定する）
    if (wasTap && !gestureMoved && Date.now() >= suppressUntil) {
      var g = nearestGid3d(c.clientX, c.clientY, 18);
      if (g === null) clearSelection(); else selectPoint(g, c.clientX, c.clientY);
    }
  }, { capture: true, passive: true });
}
// 選択のトリガ（2Dタッチ以外）は DOM の click（押下→離すで動いていない操作）:
//  (a) ホバー中の点があればその場で選択（PC はマウスが点の上で止まってからクリックされる）
//  (b) なければ、遅れて届く plotly_click / plotly_hover を少し待ってその点を選択
//      （gl3d の判定は描画フレーム後に届き、押下が短いと click 自体が出ないこともある）
//  (c) 期限内に届かなければ点のない場所として選択解除
// 古いカードは新しい判定が届くまで消さない（消えてから出る「ちらつき」を避ける）
var pendingTap = null;
function resolveTap(gid) {
  if (!pendingTap) return false;
  var t = pendingTap; clearTimeout(t.timer); pendingTap = null;
  actOn(gid, t.x, t.y);
  return true;
}
function cancelTap(clear) {
  if (!pendingTap) return;
  clearTimeout(pendingTap.timer); pendingTap = null;
  if (clear) clearSelection();
}
plot.addEventListener('click', function (e) {
  if (isTouch) return;  // タッチは 2D/3D とも自前のタップ処理
  var now = Date.now();
  if (gestureMoved || now < suppressUntil) return;
  var xy = (isTouch && now - lastTouchAt < 1000) ? lastTouchXY : [e.clientX, e.clientY];
  cancelTap(false);
  if (!isTouch && hoveredGid !== null) { openKaken(hoveredGid); return; }
  // 2D の plotly_click は DOM click より先に同期で届くので待ちは短くてよい。gl3d は描画フレーム後に届く
  pendingTap = { x: xy[0], y: xy[1], timer: setTimeout(function () { cancelTap(true); }, is3d ? 700 : 60) };
});
plot.on('plotly_click', function (d) {
  if (isTouch) return;
  var gid = gidOf(d.points[0]);
  if (gid === null) {  // 球面の下地など点以外: タップ位置の最寄りの点で補完し、なければ空白扱い
    if (pendingTap) {
      var g = nearestGid3d(pendingTap.x, pendingTap.y, isTouch ? 14 : 8);
      if (g !== null) { resolveTap(g); return; }
    }
    cancelTap(true); return;
  }
  if (Date.now() < suppressUntil || gestureMoved) return;
  resolveTap(gid);  // 待ちが無い（click より先に届いた等）場合は、直後の DOM click が開くので何もしない
});

// ---- タッチ端末共通: 地図を動かし始めたら（指が10px以上動く／2本指になる）選択カードとリングを閉じる ----
// カードは「その点を読む」ための一時的な表示で、回転・パン・ピンチ中は場所との対応も崩れるため
if (isTouch) {
  var moveStart = null;
  plot.addEventListener('touchstart', function (e) {
    moveStart = (e.touches.length === 1) ? [e.touches[0].clientX, e.touches[0].clientY] : null;
    if (e.touches.length >= 2) clearSelection();
  }, { capture: true, passive: true });
  plot.addEventListener('touchmove', function (e) {
    if (selGid === null) return;
    if (e.touches.length >= 2) { clearSelection(); return; }
    if (moveStart && (Math.abs(e.touches[0].clientX - moveStart[0]) > 10 || Math.abs(e.touches[0].clientY - moveStart[1]) > 10)) clearSelection();
  }, { capture: true, passive: true });
}

// ---- 2Dタッチ端末のタップ処理（Plotlyのタッチ経由ヒットテストは信頼できないため、
// タップ座標から最近傍の可視点を自前判定） ----
function nearestGid(cx, cy) {
  var fl = plot._fullLayout, rect = plot.getBoundingClientRect();
  var l = fl._size.l, t = fl._size.t, w = fl._size.w, h = fl._size.h;
  var px = cx - rect.left, py = cy - rect.top;
  if (px < l || px > l + w || py < t || py > t + h) return null;
  var xr = fl.xaxis.range, yr = fl.yaxis.range;
  var xppu = w / (xr[1] - xr[0]), yppu = h / (yr[1] - yr[0]);
  var xd = xr[0] + (px - l) / xppu, yd = yr[1] - (py - t) / yppu;
  var tol = 18, best = null, bd = tol * tol;
  for (var g = 0; g < M.n; g++) {
    if (!traceVisible(traceOf[g])) continue;
    var ddx = (xs[g] - xd) * xppu, ddy = (ys[g] - yd) * yppu;
    var d2 = ddx * ddx + ddy * ddy;
    if (d2 < bd) { bd = d2; best = g; }
  }
  return best;
}
function handleTap(cx, cy) {
  if (Date.now() < suppressUntil) return;
  var gid = nearestGid(cx, cy);
  if (gid === null) { clearSelection(); return; }
  selectPoint(gid, cx, cy);
}
if (isTouch && !is3d) {
  var tapStart = null;
  plot.addEventListener('touchstart', function (e) {
    tapStart = (e.touches.length === 1)
      ? { x: e.touches[0].clientX, y: e.touches[0].clientY, t: Date.now() } : null;
  }, { capture: true, passive: true });
  plot.addEventListener('touchend', function (e) {
    if (!tapStart || e.touches.length) return;
    var c = e.changedTouches[0];
    var moved = Math.abs(c.clientX - tapStart.x) > 10 || Math.abs(c.clientY - tapStart.y) > 10;
    var slow = Date.now() - tapStart.t > 500;
    tapStart = null;
    if (!moved && !slow) handleTap(c.clientX, c.clientY);
  }, { capture: true, passive: true });
}

// ---- 検索（タイトル・キーワード・課題番号の部分一致 → ハイライト＋一覧） ----
var qInput = document.getElementById('ka-q');
var qResults = document.getElementById('ka-results');
var hlIndex = null, qTimer = null;

function clearHighlight() {
  if (hlIndex !== null) { Plotly.deleteTraces(plot, hlIndex); hlIndex = null; }
  qResults.style.display = 'none'; qResults.innerHTML = '';
}
function traceVisible(ti) {
  var v = plot.data[ti].visible;
  return v === undefined || v === true;
}
// 3D/球面: カメラをその点に向けてから（球面は点が正面に来る向き、3D は注視点を点に移して寄る）、
// 描画後に点の画面座標を求めてタッチはリング＋カード。PC はカメラ移動のみ
function focusPoint3d(gid) {
  plot._lastFocus = gid;  // 検証用
  var cam = plot._fullLayout.scene.camera || {};
  var eye = cam.eye || { x: 1.25, y: 1.25, z: 1.25 }, ctr = cam.center || { x: 0, y: 0, z: 0 }, up = cam.up || { x: 0, y: 0, z: 1 };
  // Plotly のカメラ座標（eye/center）は、データ座標 × dataScale にシーンの model 行列（アスペクト比の拡縮＋箱の中心への
  // 平行移動）を掛けた空間で表される。model を掛け忘れると 3D（箱が原点対称でない）で注視点がずれる
  var sc = plot._fullLayout.scene._scene, ds = (sc && sc.dataScale) || [1, 1, 1];
  var sx = xs[gid] * ds[0], sy = ys[gid] * ds[1], sz = zs[gid] * ds[2];
  var mm = sc && sc.glplot && sc.glplot.cameraParams && sc.glplot.cameraParams.model;
  var px = sx, py = sy, pz = sz, upd;
  if (mm) {
    px = mm[0] * sx + mm[4] * sy + mm[8] * sz + mm[12];
    py = mm[1] * sx + mm[5] * sy + mm[9] * sz + mm[13];
    pz = mm[2] * sx + mm[6] * sy + mm[10] * sz + mm[14];
  }
  if (isGlobe) {
    var n = Math.hypot(px, py, pz) || 1, d = Math.hypot(eye.x, eye.y, eye.z);
    var ux = px / n, uy = py / n, uz = pz / n;
    var upv = Math.abs(uz) > 0.9 ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };  // 極付近では上向きを差し替え
    upd = { 'scene.camera.eye': { x: ux * d, y: uy * d, z: uz * d }, 'scene.camera.center': { x: 0, y: 0, z: 0 }, 'scene.camera.up': upv };
  } else {
    var dx = eye.x - ctr.x, dy = eye.y - ctr.y, dz = eye.z - ctr.z, len = Math.hypot(dx, dy, dz) || 1;
    var dist = Math.max(0.5, Math.min(len, 0.9));  // 近づきすぎない範囲で寄る
    upd = { 'scene.camera.center': { x: px, y: py, z: pz },
            'scene.camera.eye': { x: px + dx / len * dist, y: py + dy / len * dist, z: pz + dz / len * dist }, 'scene.camera.up': up };
  }
  Plotly.relayout(plot, upd).then(function () {
    if (!isTouch) return;
    requestAnimationFrame(function () { requestAnimationFrame(function () {
      var xy = project3d(gid) || [window.innerWidth / 2, window.innerHeight / 2];
      selectPoint(gid, xy[0], xy[1]);
    }); });
  });
}
function runSearch(q) {
  clearHighlight();
  q = q.trim().toLowerCase();
  if (q.length < 2) return;
  var hits = [];
  for (var gid = 0; gid < M.n && hits.length < 2000; gid++) {
    var ti = traceOf[gid];
    if (!traceVisible(ti)) continue;
    var row = getRow(gid);
    if (!row) continue;
    var tr = M.traces[ti];
    var hay = (row[2] + '、' + row[3] + '、' + row[0] + '、' + catOf(gid, row)).toLowerCase();
    if (hay.indexOf(q) < 0) continue;
    hits.push({ gid: gid, row: row, tr: tr, cat: catOf(gid, row) });
  }
  if (!hits.length) {
    qResults.innerHTML = '<span style="color:' + MUTED + '">該当なし</span>';
    qResults.style.display = 'block';
    return;
  }
  var overlay = {
    x: hits.map(function (h) { return xs[h.gid]; }),
    y: hits.map(function (h) { return ys[h.gid]; }),
    mode: 'markers', hoverinfo: 'none', showlegend: false,
    marker: { size: is3d ? 4 : 9, color: 'rgba(11,11,11,0)',
              line: { width: 2, color: INK } },
    type: is3d ? 'scatter3d' : 'scattergl',
  };
  if (is3d) overlay.z = hits.map(function (h) { return zs[h.gid]; });
  Plotly.addTraces(plot, overlay).then(function () { hlIndex = plot.data.length - 1; });
  var html = '<b>' + fmt(hits.length) + (hits.length >= 2000 ? '+' : '') +
    '件ヒット</b><span style="color:' + MUTED + '">（先頭30件）</span><br>';
  hits.slice(0, 30).forEach(function (h, k) {
    html += '<a href="#" class="ka-hit" data-k="' + k + '" style="display:block;' +
      'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#1c5cab;' +
      'text-decoration:none;padding:1px 0">' + esc((h.row[2] || '').slice(0, 48)) +
      ' <span style="color:' + MUTED + '">' + esc(h.cat + ' / ' + h.row[0]) +
      '</span></a>';
  });
  qResults.innerHTML = html;
  qResults.style.display = 'block';
  qResults.querySelectorAll('.ka-hit').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      var h = hits[parseInt(a.getAttribute('data-k'), 10)];
      if (isTouch) qResults.style.display = 'none';  // スマホは結果窓が地図を覆うので閉じる（強調は残す）
      if (is3d) { focusPoint3d(h.gid); return; }
      var span = 1.5;
      Plotly.relayout(plot, {
        'xaxis.range': [xs[h.gid] - span, xs[h.gid] + span],
        'yaxis.range': [ys[h.gid] - span, ys[h.gid] + span],
      }).then(function () {  // タッチはズーム後にカード。PC は検索ヒットの輪郭表示が場所を示す
        if (isTouch) { var xy = projected(h.gid); selectPoint(h.gid, xy[0], xy[1]); }
      });
    });
  });
}
// 入力欄右端の × : 検索語・強調・結果窓をすべて消す。結果窓の外をクリック/タップ: 結果窓だけ閉じる
// （地図上の強調は残す）。入力欄に戻れば結果窓を再表示する（2026-09-09 ユーザ仕様）
var qClear = document.getElementById('ka-q-clear');
function updateClearBtn() { qClear.style.display = qInput.value ? 'block' : 'none'; }
function resetSearch() { qInput.value = ''; clearHighlight(); updateClearBtn(); }
qClear.addEventListener('click', function (e) { e.preventDefault(); resetSearch(); if (!isTouch) qInput.focus(); });
document.addEventListener('pointerdown', function (e) {
  if (qResults.style.display === 'none') return;
  var wrap = document.getElementById('ka-q-wrap');
  if (!wrap.contains(e.target)) qResults.style.display = 'none';
}, true);
qInput.addEventListener('focus', function () {
  if (qInput.value.trim().length >= 2 && qResults.innerHTML) qResults.style.display = 'block';
});
qInput.addEventListener('input', function () {
  updateClearBtn();
  if (qTimer) clearTimeout(qTimer);
  qTimer = setTimeout(function () { runSearch(qInput.value); }, 300);
});
qInput.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') { resetSearch(); qInput.blur(); e.stopPropagation(); }
});

// ---- 種目フィルタ ----
var counts = {};
M.traces.forEach(function (t) {
  if (t.k === 'd') counts[t.cat] = (counts[t.cat] || 0) + t.n;
});
var order = M.catOrder;
var cats = order.filter(function (c) { return counts[c] !== undefined; }).concat(
  Object.keys(counts).filter(function (c) { return order.indexOf(c) < 0; })
    .sort(function (a, b) { return counts[b] - counts[a]; }));

document.getElementById('ka-body').innerHTML =
  '<a href="#" id="ka-selall" style="color:#1c5cab;text-decoration:none">全選択</a>&nbsp; ' +
  '<a href="#" id="ka-selnone" style="color:#1c5cab;text-decoration:none">全解除</a>' +
  cats.map(function (c) {
    return '<label style="display:block;white-space:nowrap;cursor:pointer">' +
      '<input type="checkbox" class="ka-cat" checked style="vertical-align:-2px"> ' +
      esc(c) + ' <span style="color:' + MUTED + '">' + fmt(counts[c]) + '</span></label>';
  }).join('');
var boxes = document.querySelectorAll('.ka-cat');

function setCategory(cat, on) {
  var idx = [], vis = [];
  plot.data.forEach(function (t, i) {
    if (t.meta !== cat) return;
    if (!on) { idx.push(i); vis.push(false); return; }
    var v = true;
    plot.data.some(function (s) {
      if (s.legendgroup === t.legendgroup && s.visible !== false) {
        v = (s.visible === undefined) ? true : s.visible;
        return true;
      }
      return false;
    });
    idx.push(i); vis.push(v);
  });
  if (idx.length) Plotly.restyle(plot, { visible: vis }, idx);
}
boxes.forEach(function (cb, i) {
  cb.addEventListener('change', function () { setCategory(cats[i], cb.checked); });
});
document.getElementById('ka-selall').addEventListener('click', function (e) {
  e.preventDefault();
  boxes.forEach(function (cb, i) { cb.checked = true; setCategory(cats[i], true); });
});
document.getElementById('ka-selnone').addEventListener('click', function (e) {
  e.preventDefault();
  boxes.forEach(function (cb, i) { cb.checked = false; setCategory(cats[i], false); });
});

// ---- 選択パネル（なげなわ/矩形で囲うと内訳・キーワード集計を即時表示） ----
var selPanel = document.createElement('div');
selPanel.style.cssText = 'position:fixed;bottom:14px;left:14px;z-index:999;display:none;' +
  'max-width:400px;max-height:55vh;overflow-y:auto;padding:10px 14px;' + PANEL;
document.body.appendChild(selPanel);

function topEntries(obj, n) {
  return Object.keys(obj).sort(function (a, b) { return obj[b] - obj[a]; }).slice(0, n);
}
var outlineClearedAt = 0;
plot.on('plotly_selected', function (d) {
  if (!d || !d.points || !d.points.length) {
    if (Date.now() - outlineClearedAt < 800) return;
    selPanel.style.display = 'none'; return;
  }
  var n = d.points.length, dais = {}, cts = {}, kws = {}, missing = 0;
  d.points.forEach(function (p) {
    var gid = gidOf(p);
    if (gid === null) return;
    var tr = M.traces[traceOf[gid]];
    dais[tr.dai] = (dais[tr.dai] || 0) + 1;
    cts[tr.cat] = (cts[tr.cat] || 0) + 1;
    var row = getRow(gid);
    if (!row) { missing++; return; }
    if (row[3]) {
      row[3].split('、').forEach(function (w) { if (w) kws[w] = (kws[w] || 0) + 1; });
    }
  });
  var html = '<b>選択: ' + fmt(n) + '件</b>' +
    ' <a href="#" id="ka-selclear" style="color:' + MUTED + '">閉じる</a><br>';
  html += '<span style="color:' + SUB + '">大区分:</span> ' + topEntries(dais, 5).map(function (g) {
    return esc(g) + ' ' + fmt(dais[g]);
  }).join(' / ') + '<br>';
  html += '<span style="color:' + SUB + '">種目:</span> ' + topEntries(cts, 4).map(function (c) {
    return esc(c) + ' ' + fmt(cts[c]);
  }).join(' / ') + '<br>';
  html += '<span style="color:' + SUB + '">頻出キーワード:</span>' +
    (missing ? ' <span style="color:' + MUTED + '">（読み込み中の' + fmt(missing) +
      '件は集計外）</span>' : '') + '<br>' +
    topEntries(kws, 15).map(function (w) {
      return '<span style="display:inline-block;background:#eef3fa;border:1px solid #c9d8ee;' +
        'border-radius:4px;padding:0 6px;margin:1px 2px">' + esc(w) +
        ' <span style="color:' + MUTED + '">' + kws[w] + '</span></span>';
    }).join('');
  selPanel.innerHTML = html;
  selPanel.style.display = 'block';
  document.getElementById('ka-selclear').addEventListener('click', function (e) {
    e.preventDefault(); selPanel.style.display = 'none';
  });
  outlineClearedAt = Date.now();
  Plotly.relayout(plot, { selections: [] });
});
plot.on('plotly_deselect', function () {
  if (Date.now() - outlineClearedAt < 800) return;
  selPanel.style.display = 'none';
});

// Esc: 選択を解除してパン操作モードに戻る
document.addEventListener('keydown', function (e) {
  if (e.key !== 'Escape') return;
  selPanel.style.display = 'none';
  clearSelection();
  Plotly.update(plot, { selectedpoints: null }, { selections: [], dragmode: 'pan' });
});

// ---- 大区分シート（スマホ幅のみ。Plotly凡例の代替: 下部ボタン→ボトムシート） ----
if (narrow) {
  var anchors = M.traces.filter(function (t) { return t.k === 'a'; })
    .sort(function (a, b) { return a.rank - b.rank; });  // 凡例順（A〜K→複数→区分なし）
  var daiOn = {};
  anchors.forEach(function (t) { daiOn[t.dai] = t.vis; });

  // 表示状態 daiOn をまとめて反映（1回の restyle。3D では restyle ごとに再アップロードが走るため）
  function applyDai() {
    var idx = [], vis = [];
    plot.data.forEach(function (t, i) {
      if (!t.legendgroup || !(t.legendgroup in daiOn)) return;
      var on = daiOn[t.legendgroup];
      if (on) { if (t.visible === 'legendonly') { idx.push(i); vis.push(true); } }
      else if (t.visible !== false) { idx.push(i); vis.push('legendonly'); }
    });
    if (idx.length) Plotly.restyle(plot, { visible: vis }, idx);
    drawer.querySelectorAll('.ka-dai-item').forEach(function (el) {
      el.style.opacity = daiOn[anchors[parseInt(el.getAttribute('data-i'), 10)].dai] ? '' : '0.35';
    });
  }

  var daiBtn = document.getElementById('ka-dai-btn');
  document.getElementById('ka-dai-wrap').style.display = 'flex';

  var backdrop = document.createElement('div');
  backdrop.id = 'ka-dai-backdrop';
  backdrop.style.cssText = 'position:fixed;inset:0;z-index:1000;background:rgba(20,20,15,0.3);' +
    'opacity:0;visibility:hidden;transition:opacity .25s';
  document.body.appendChild(backdrop);

  var drawer = document.createElement('div');
  drawer.id = 'ka-dai-drawer';
  drawer.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:1001;' +
    'background:#fcfcfb;border-top:1px solid ' + LINE + ';border-radius:16px 16px 0 0;' +
    'box-shadow:0 -6px 24px rgba(0,0,0,0.18);transform:translateY(105%);' +
    'transition:transform .28s cubic-bezier(.2,.8,.25,1);' +
    'padding:10px 16px calc(16px + env(safe-area-inset-bottom));max-height:72vh;overflow-y:auto;' +
    'font:12.5px/1.6 -apple-system,sans-serif;color:' + INK;
  var BTN = 'padding:4px 10px;border:1px solid ' + LINE + ';border-radius:12px;background:#fff;color:#1c5cab;font-size:11.5px';
  drawer.innerHTML =
    '<div style="width:36px;height:4px;border-radius:2px;background:#d5d4cc;margin:0 auto 10px"></div>' +
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px">' +
    '<b style="font-size:14px">大区分</b>' +
    '<span style="color:' + MUTED + ';font-size:11px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">タップ: 表示切替</span>' +
    '<span id="ka-dai-all" style="' + BTN + '">すべて表示</span>' +
    '<span id="ka-dai-none" style="' + BTN + '">すべて非表示</span></div>' +
    '<div style="display:grid;grid-template-columns:1fr;gap:5px">' +
    anchors.map(function (t, i) {
      return '<div class="ka-dai-item" data-i="' + i + '" style="display:flex;align-items:center;' +
        'min-width:0;gap:8px;border:1px solid ' + LINE + ';border-radius:9px;padding:5px 6px 5px 10px;' +
        'background:#fff;transition:opacity .15s;' + (t.vis ? '' : 'opacity:0.35') + '">' +
        '<span style="flex:none;width:11px;height:11px;border-radius:50%;background:' + t.color + '"></span>' +
        '<span style="flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' +
        esc(t.label) + '</span>' +
        '<span style="color:' + MUTED + ';font-size:11px">' + fmt(t.n) + '</span>' +
        '<span class="ka-dai-only" data-i="' + i + '" style="flex:none;padding:2px 7px;border-radius:9px;' +
        'background:#eef2f8;color:#1c5cab;font-size:11px">のみ</span></div>';
    }).join('') + '</div>';
  document.body.appendChild(drawer);

  function openDrawer(on) {
    drawer.style.transform = on ? 'translateY(0)' : 'translateY(105%)';
    drawer.dataset.open = on ? '1' : '';
    backdrop.style.opacity = on ? '1' : '0';
    backdrop.style.visibility = on ? 'visible' : 'hidden';
  }
  daiBtn.addEventListener('click', function () { clearSelection(); openDrawer(true); });
  backdrop.addEventListener('click', function () { openDrawer(false); });
  drawer.querySelectorAll('.ka-dai-item').forEach(function (el) {
    el.addEventListener('click', function () {
      var t = anchors[parseInt(el.getAttribute('data-i'), 10)];
      daiOn[t.dai] = !daiOn[t.dai];
      applyDai();
    });
  });
  drawer.querySelectorAll('.ka-dai-only').forEach(function (el) {  // その大区分だけ表示
    el.addEventListener('click', function (e) {
      e.stopPropagation();
      var t = anchors[parseInt(el.getAttribute('data-i'), 10)];
      anchors.forEach(function (a) { daiOn[a.dai] = a.dai === t.dai; });
      applyDai();
    });
  });
  document.getElementById('ka-dai-all').addEventListener('click', function () {
    anchors.forEach(function (a) { daiOn[a.dai] = true; }); applyDai();
  });
  document.getElementById('ka-dai-none').addEventListener('click', function () {
    anchors.forEach(function (a) { daiOn[a.dai] = false; }); applyDai();
  });
}

// ---- 2本指ピンチズーム（2D・タッチ端末のみ。Plotly 2Dカルテシアンは
// タッチのピンチに対応していないため自前実装。1本指パンはPlotly標準） ----
function tDist(t) {
  var dx = t[0].clientX - t[1].clientX, dy = t[0].clientY - t[1].clientY;
  return Math.sqrt(dx * dx + dy * dy);
}
if (isTouch && !is3d) {
  var pinch = null, pinchRaf = false;
  plot.addEventListener('touchstart', function (e) {
    if (e.touches.length !== 2) return;
    e.stopPropagation();  // Plotlyのパン処理に2本指を渡さない
    var fl = plot._fullLayout;
    pinch = {
      d0: tDist(e.touches),
      cx: (e.touches[0].clientX + e.touches[1].clientX) / 2,
      cy: (e.touches[0].clientY + e.touches[1].clientY) / 2,
      xr: fl.xaxis.range.slice(), yr: fl.yaxis.range.slice(),
      rect: plot.getBoundingClientRect(),
      sz: { l: fl._size.l, t: fl._size.t, w: fl._size.w, h: fl._size.h },
    };
  }, { capture: true, passive: true });
  plot.addEventListener('touchmove', function (e) {
    if (!pinch || e.touches.length !== 2) return;
    e.preventDefault(); e.stopPropagation();
    if (pinchRaf) return;
    pinchRaf = true;
    var s = pinch.d0 / tDist(e.touches);  // 指を広げる=s<1=ズームイン
    requestAnimationFrame(function () {
      pinchRaf = false;
      if (!pinch) return;
      var fx = (pinch.cx - pinch.rect.left - pinch.sz.l) / pinch.sz.w;
      var fy = (pinch.cy - pinch.rect.top - pinch.sz.t) / pinch.sz.h;
      var xc = pinch.xr[0] + (pinch.xr[1] - pinch.xr[0]) * fx;
      var yc = pinch.yr[1] - (pinch.yr[1] - pinch.yr[0]) * fy;  // 画面yは下向き
      Plotly.relayout(plot, {
        'xaxis.range': [xc - (xc - pinch.xr[0]) * s, xc + (pinch.xr[1] - xc) * s],
        'yaxis.range': [yc - (yc - pinch.yr[0]) * s, yc + (pinch.yr[1] - yc) * s],
      });
    });
  }, { capture: true, passive: false });
  plot.addEventListener('touchend', function (e) {
    if (e.touches.length < 2) pinch = null;
  }, { capture: true, passive: true });
}

// ---- 3D/球面のタッチ操作（自前のカメラ制御）----
// Plotly の 3D はタッチの回転終了時に内部状態からカメラを書き戻すことがあり、こちらの relayout と
// 競合して「ピンチで初期の向きに戻る」「離すと前の位置に戻る」が起きた。タッチ端末では Plotly に
// タッチイベントを一切渡さず（capture で stopPropagation）、1本指=回転、2本指=拡大縮小（+3Dは移動）を
// すべて自前のカメラ状態から計算して relayout で渡す。
function v3(x, y, z) { return { x: x, y: y, z: z }; }
function vsub(a, b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
function vadd(a, b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
function vmul(a, k) { return v3(a.x * k, a.y * k, a.z * k); }
function vdot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
function vcross(a, b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
function vlen(a) { return Math.sqrt(vdot(a, a)); }
function vnorm(a) { var l = vlen(a) || 1; return vmul(a, 1 / l); }
function vrot(v, axis, ang) {  // ロドリゲスの回転公式（axis は単位ベクトル）
  var c = Math.cos(ang), sn = Math.sin(ang);
  return vadd(vadd(vmul(v, c), vmul(vcross(axis, v), sn)), vmul(axis, vdot(axis, v) * (1 - c)));
}
function tCenter(t) { return [(t[0].clientX + t[1].clientX) / 2, (t[0].clientY + t[1].clientY) / 2]; }
if (isTouch && is3d) {
  var g3 = null, raf3 = false, pendingCam = null, lastCam = null;
  // 慣性（3D・球面）: 1本指回転の直近の指速度を離した後も減衰させながら回し続ける。
  // 次のタッチ開始で止める。速度は touchend 直前 ~150ms の touchmove から求める
  // （指を止めてから離した場合は速度ゼロ→慣性なし）。
  var INERTIA = true, inert = null, samples = [];
  function sampleTouch(t, x, y) {
    samples.push({ t: t, x: x, y: y });
    // 直近 ~150ms、ただし描画が重くイベントが間引かれても速度が取れるよう最低3点は残す
    while (samples.length > 3 && t - samples[0].t > 150) samples.shift();
  }
  function stepInertia(now) {
    if (!inert) return;
    var dt = Math.min(50, now - inert.t); inert.t = now;
    var c = inert.cam, bs = basis(c);
    var f1 = vrot(vsub(c.eye, c.ctr), bs.u, -inert.wx * dt), up1 = c.up;
    f1 = vrot(f1, bs.r, -inert.wy * dt); up1 = vrot(up1, bs.r, -inert.wy * dt);
    c = { eye: vadd(c.ctr, f1), ctr: c.ctr, up: up1 };
    inert.cam = c; applyCam(c);
    var decay = Math.exp(-dt / 500);  // 時定数 500ms（60fps なら約 2〜3 秒で止まる）
    inert.wx *= decay; inert.wy *= decay;
    if (Math.hypot(inert.wx, inert.wy) < 2e-5) { inert = null; return; }
    requestAnimationFrame(stepInertia);
  }
  function startInertia(endTime) {
    if (!INERTIA || samples.length < 2) return;
    var a = samples[0], b = samples[samples.length - 1], dt = b.t - a.t;
    if (dt <= 0 || dt > 400 || endTime - b.t > 100) return;  // 止めてから離した／描画が重く速度が測れない
    var vx = (b.x - a.x) / dt, vy = (b.y - a.y) / dt, sp = Math.hypot(vx, vy);  // px/ms
    if (sp < 0.05) return;
    if (sp > 2) { vx *= 2 / sp; vy *= 2 / sp; }
    // 指速度の半分から始める（全速だと最大 1 周近く回ってしまう。上限 2px/ms で約 0.6 周、1px/ms で約 1/3 周）
    var k = 0.5 * Math.PI / Math.max(plot._fullLayout._size.w, 1);
    inert = { wx: vx * k, wy: vy * k, t: performance.now(), cam: lastCam || liveCam() };
    requestAnimationFrame(stepInertia);
  }
  function liveCam() {  // 実際のカメラ（レイアウト上の値は更新が遅れることがある）
    var sc = plot._fullLayout.scene && plot._fullLayout.scene._scene;
    var c = (sc && sc.getCamera) ? sc.getCamera() : (plot._fullLayout.scene.camera || {});
    var eye = c.eye || v3(1.25, 1.25, 1.25), ctr = c.center || v3(0, 0, 0), up = c.up || v3(0, 0, 1);
    return { eye: v3(eye.x, eye.y, eye.z), ctr: v3(ctr.x, ctr.y, ctr.z), up: v3(up.x, up.y, up.z) };
  }
  function basis(c) {  // 視線 f、画面右 r、画面上 u
    var f = vnorm(vsub(c.ctr, c.eye));
    var r = vnorm(vcross(f, c.up));
    return { f: f, r: r, u: vcross(r, f) };
  }
  function applyCam(c) {
    lastCam = c;
    pendingCam = c;
    if (raf3) return;
    raf3 = true;
    requestAnimationFrame(function () {
      raf3 = false;
      if (!pendingCam) return;
      var pc = pendingCam; pendingCam = null;
      Plotly.relayout(plot, { 'scene.camera.eye': pc.eye, 'scene.camera.center': pc.ctr, 'scene.camera.up': pc.up });
    });
  }
  function startGesture(touches) {
    samples = [];
    // 慣性で回転中は relayout が追いつかず liveCam が古いことがあるので、自前の最新カメラを使う
    var c = inert ? inert.cam : liveCam(), bs = basis(c);
    inert = null;
    if (touches.length === 1) {
      g3 = { mode: 'rot', x: touches[0].clientX, y: touches[0].clientY, cam: c, bs: bs,
             k: Math.PI / Math.max(plot._fullLayout._size.w, 1) };  // 画面幅ぶんのドラッグで半回転（180°）
    } else if (touches.length === 2) {
      var c0 = tCenter(touches);
      g3 = { mode: 'two', d0: tDist(touches), cx: c0[0], cy: c0[1], cam: c, bs: bs,
             // 1ピクセルあたりの空間距離（透視投影 fovy=45° で注視点距離の画面高さから換算）
             k: 2 * vlen(vsub(c.eye, c.ctr)) * Math.tan(Math.PI / 8) / plot._fullLayout._size.h };
    } else g3 = null;
  }
  plot.addEventListener('touchstart', function (e) {
    e.stopPropagation();
    startGesture(e.touches);
  }, { capture: true, passive: true });
  plot.addEventListener('touchmove', function (e) {
    e.preventDefault(); e.stopPropagation();
    // 指の本数がジェスチャ開始時と違えば（2本→1本など）現在の指で始め直す
    if (!g3 || (g3.mode === 'rot') !== (e.touches.length === 1)) startGesture(e.touches);
    if (!g3) return;
    var c = g3.cam, bs = g3.bs;
    if (g3.mode === 'rot' && e.touches.length === 1) {
      // 指の動きに場面が付いてくる向き: 右へ動かす→場面が右へ→カメラは上方向軸まわりに逆回転。
      // 自由回転（固定軸なし）: 回転軸はジェスチャ開始時のカメラの上・右ベクトル
      var dx = e.touches[0].clientX - g3.x, dy = e.touches[0].clientY - g3.y;
      sampleTouch(e.timeStamp, e.touches[0].clientX, e.touches[0].clientY);
      var f = vsub(c.eye, c.ctr);
      var f1 = vrot(f, bs.u, -dx * g3.k), up1 = c.up;
      f1 = vrot(f1, bs.r, -dy * g3.k); up1 = vrot(up1, bs.r, -dy * g3.k);
      applyCam({ eye: vadd(c.ctr, f1), ctr: c.ctr, up: up1 });
    } else if (g3.mode === 'two' && e.touches.length === 2) {
      var s = Math.max(0.05, g3.d0 / tDist(e.touches));  // 指を広げる=s<1=近づく
      var cc = tCenter(e.touches), mx = cc[0] - g3.cx, my = cc[1] - g3.cy;
      // 球面は並行移動なし（球の中心を固定し、拡大縮小のみ）。3D は重心の移動で視点・注視点を平行移動
      var T = isGlobe ? v3(0, 0, 0) : vadd(vmul(bs.r, -mx * g3.k), vmul(bs.u, my * g3.k));
      applyCam({ eye: vadd(vadd(c.ctr, vmul(vsub(c.eye, c.ctr), s)), T), ctr: vadd(c.ctr, T), up: c.up });
    }
  }, { capture: true, passive: false });
  plot.addEventListener('touchend', function (e) {
    e.stopPropagation();
    if (e.touches.length > 0) startGesture(e.touches);  // 2本→1本: 残った指で回転を続ける
    else {
      if (g3 && g3.mode === 'rot') startInertia(e.timeStamp);
      g3 = null;
    }
  }, { capture: true, passive: true });
  plot.addEventListener('touchcancel', function () { g3 = null; samples = []; }, { capture: true, passive: true });
}

}  // setupUI
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
