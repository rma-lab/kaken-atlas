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


def build_order(df: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    """plot_map_interactive.py と同じ描画順に並べ、トレース表を作る。

    点は「大区分（下層→上層）×種目（件数降順）」でトレースごとに連続配置。
    グローバル添字＝この並びの行番号がシャード分割の単位になる。
    """
    draw_order = ["区分なし", "複数", *DAI_COLORS.keys()]
    legend_order = [*DAI_COLORS.keys(), "複数", "区分なし"]
    parts: list[pl.DataFrame] = []
    traces: list[dict] = []
    offset = 0
    for dai in draw_order:
        dsub = df.filter(pl.col("dai") == dai)
        if dsub.height == 0:
            continue
        color = DAI_COLORS.get(dai, "#b9b8b0")
        label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
        visible = dai != "区分なし"
        traces.append(dict(
            k="a", dai=dai, label=label, color=color, n=dsub.height,
            rank=legend_order.index(dai) + 1, vis=visible,
        ))
        # len 同数の種目間の順序を固定するため category 名でタイブレーク（出力の再現性）
        cat_counts = dsub.group_by("category").len().sort(
            ["len", "category"], descending=[True, False]
        )
        for cat in cat_counts["category"]:
            sub = dsub.filter(pl.col("category") == cat)
            traces.append(dict(
                k="d", dai=dai, label=label, color=color, cat=cat,
                off=offset, n=sub.height, vis=visible,
            ))
            parts.append(sub)
            offset += sub.height
    return pl.concat(parts), traces


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

    big, traces = build_order(df)
    n = big.height
    dims = ["c0", "c1"] + (["c2"] if is_3d else [])
    if is_globe:
        # 全点が同じ半径だと、重なりの勝者が描画順（最後の大区分）で決まって色が偏る。
        # 半径に微小な乱数を与え、奥行きテストで「一番外側の点」が見えるようにする（再現性のため seed 固定）
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
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "points.bin").write_bytes(points_bin)

    ktype_idx = {t: i for i, t in enumerate(ktypes)}
    rows = list(zip(
        big["award_number"],
        (ktype_idx[t] for t in big["ktype"]),
        big["title"],
        ("、".join(kw) if kw is not None else "" for kw in big["keywords"]),
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
        pointsBytes=len(points_bin), quant=quant, ktypes=ktypes,
        traces=traces, catOrder=CATEGORY_ORDER,
        footer=FOOTER + (f" | 球面埋め込み: output_metric=haversine{globe_params(coords_path)}" if is_globe else ""),
        title=f"科研費 学術地図 {'球面' if is_globe else ('3D' if is_3d else '2D')}",
        sub=f"2019–2025年度・{n:,}件",
    )
    html = (
        TEMPLATE
        .replace("__TITLE__", manifest["title"])
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
        .replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False,
                                            separators=(",", ":")))
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"出力: {out_dir}/ 合計 {total / 1e6:.1f} MB "
          f"(points.bin {len(points_bin) / 1e6:.1f} MB, シャード {n_shards} 個, "
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
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🗺️</text></svg>">
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
      marker: is3d ? { size: isGlobe ? 1.6 : 1.3, color: t.color, opacity: isGlobe ? 1 : 0.55 }
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
    if (isGlobe) {  // 球全体が収まる距離から見る。北極(z)を上に
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
    { scrollZoom: true, displaylogo: false, doubleClick: 'reset', responsive: true });

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
bar.innerHTML =
  '<div id="ka-title" style="white-space:nowrap"><b style="font-size:14.5px">' + M.title + '</b>' +
  ' <span class="ka-sub" style="color:' + MUTED + ';font-size:12px">' + M.sub + '</span></div>' +
  '<div style="position:relative;flex:0 1 300px;min-width:170px">' +
  '  <input id="ka-q" type="search" placeholder="詳細データ読み込み中…" disabled' +
  '   style="width:100%;box-sizing:border-box;padding:6px 12px;border:1px solid #cfcec7;' +
  '   border-radius:15px;background:#fff;font:12.5px -apple-system,sans-serif;outline:none">' +
  '  <div id="ka-results" style="display:none;position:absolute;top:36px;left:0;width:380px;' +
  '   max-height:55vh;overflow-y:auto;padding:8px 12px;' + PANEL + '"></div>' +
  '</div>' +
  '<div style="flex:1"></div>' +
  '<div id="ka-filter-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span id="ka-filter-btn" style="cursor:default;color:' + SUB + '">種目フィルタ ▾</span>' +
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
var menus = [['ka-filter-wrap', 'ka-body', 'ka-filter-btn'],
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

document.getElementById('ka-help-body').innerHTML = isTouch
  ? (is3d ? '<div>1本指: 回転 / 2本指ピンチ: 拡大縮小 / 2本指ドラッグ: 移動</div>'
          : '<div>1本指: 移動 / 2本指ピンチ: 拡大縮小</div>' +
            '<div>ダブルタップ: 全体表示に戻る</div>') +
    '<div>点をタップ: 詳細カード / カードをタップ: KAKENページ</div>' +
    '<div>凡例タップ: 大区分の表示切替</div>'
  : is3d
  ? '<div>ドラッグ: 回転 / スクロール: 拡大縮小</div>' +
    '<div>点にホバー: 概要 / クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替</div>'
  : '<div>スクロール: 拡大縮小 / ドラッグ: 移動</div>' +
    '<div>ダブルクリック: 全体表示に戻る</div>' +
    '<div>点にホバー: 概要 / クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替</div>' +
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
  var row = getRow(gid);
  var title = row ? esc((row[2] || '（タイトルなし）').slice(0, 48))
                  : '<span style="color:' + MUTED + '">（読み込み中…）</span>';
  var tail = row ? esc(tr.cat + ' / ' + row[0]) : esc(tr.cat);
  tip.innerHTML = headerHtml(tr, false) +
    '<div style="' + ELL + '">' + title + '</div>' +
    '<div style="' + ELL + '">' + tail + '</div>' +
    '<div style="color:' + MUTED + ';font-size:11px">クリックでKAKENページを開く</div>';
  tip.style.borderColor = tr.color;
  tip.style.display = 'block'; placeTip();
}

plot.on('plotly_hover', function (d) {
  var gid = gidOf(d.points[0]);
  if (gid === null) return;
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
plot.on('plotly_unhover', function () { hoveredGid = null; tip.style.display = 'none'; });

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
    '<div style="' + ELL + ';color:' + SUB + '">' + esc(row ? tr.cat + ' / ' + row[0] : tr.cat) + '</div>' +
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
    if (gestureMoved) hideRing();  // 回転すると点の画面位置が変わるのでリングは消す（カードは残す）
    down3 = null;
  }
  plot.addEventListener('mousedown', function (e) { down3 = [e.clientX, e.clientY]; }, true);
  plot.addEventListener('mouseup', function (e) { down3End(e.clientX, e.clientY); }, true);
  plot.addEventListener('touchstart', function (e) {
    down3 = (e.touches.length === 1) ? [e.touches[0].clientX, e.touches[0].clientY] : null;
  }, { capture: true, passive: true });
  plot.addEventListener('touchend', function (e) {
    if (e.touches.length || !e.changedTouches.length) return;
    var c = e.changedTouches[0];
    down3End(c.clientX, c.clientY);
    lastTouchXY = [c.clientX, c.clientY]; lastTouchAt = Date.now();
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
  if (isTouch && !is3d) return;  // 2Dタッチは自前のタップ処理（下記）
  var now = Date.now();
  if (gestureMoved || now < suppressUntil) return;
  var xy = (isTouch && now - lastTouchAt < 1000) ? lastTouchXY : [e.clientX, e.clientY];
  cancelTap(false);
  if (!isTouch && hoveredGid !== null) { openKaken(hoveredGid); return; }
  // 2D の plotly_click は DOM click より先に同期で届くので待ちは短くてよい。gl3d は描画フレーム後に届く
  pendingTap = { x: xy[0], y: xy[1], timer: setTimeout(function () { cancelTap(true); }, is3d ? 700 : 60) };
});
plot.on('plotly_click', function (d) {
  if (isTouch && !is3d) return;
  var gid = gidOf(d.points[0]);
  if (gid === null) { cancelTap(true); return; }  // 球面の下地など点以外 → 空白扱い
  if (Date.now() < suppressUntil || gestureMoved) return;
  if (resolveTap(gid)) return;
  // 待ちが無い（click より先に届いた等）場合: タッチはその点を選択。PC は直後の DOM click が開くので何もしない
  if (isTouch && selGid !== gid && Date.now() - lastTouchAt < 1000) selectPoint(gid, lastTouchXY[0], lastTouchXY[1]);
});

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
  qResults.style.display = 'none';
}
function traceVisible(ti) {
  var v = plot.data[ti].visible;
  return v === undefined || v === true;
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
    var hay = (row[2] + '、' + row[3] + '、' + row[0] + '、' + tr.cat).toLowerCase();
    if (hay.indexOf(q) < 0) continue;
    hits.push({ gid: gid, row: row, tr: tr });
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
      ' <span style="color:' + MUTED + '">' + esc(h.tr.cat + ' / ' + h.row[0]) +
      '</span></a>';
  });
  qResults.innerHTML = html;
  qResults.style.display = 'block';
  qResults.querySelectorAll('.ka-hit').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      var h = hits[parseInt(a.getAttribute('data-k'), 10)];
      if (is3d) {  // 3Dは画面位置が取れない。タッチは中央基準でカード（リングで場所を示す）、PC は何もしない
        if (isTouch) selectPoint(h.gid, window.innerWidth / 2, window.innerHeight / 2);
        return;
      }
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
qInput.addEventListener('input', function () {
  if (qTimer) clearTimeout(qTimer);
  qTimer = setTimeout(function () { runSearch(qInput.value); }, 300);
});
qInput.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') { qInput.value = ''; clearHighlight(); qInput.blur(); e.stopPropagation(); }
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

  function setDai(dai, on) {
    var idx = [], vis = [];
    plot.data.forEach(function (t, i) {
      if (t.legendgroup !== dai) return;
      if (on) { if (t.visible === 'legendonly') { idx.push(i); vis.push(true); } }
      else if (t.visible !== false) { idx.push(i); vis.push('legendonly'); }
    });
    if (idx.length) Plotly.restyle(plot, { visible: vis }, idx);
  }

  var daiBtn = document.createElement('div');
  daiBtn.id = 'ka-dai-btn';
  daiBtn.textContent = '大区分 ▴';
  daiBtn.style.cssText = 'position:fixed;bottom:calc(12px + env(safe-area-inset-bottom));' +
    'left:50%;transform:translateX(-50%);z-index:999;padding:7px 20px;border-radius:18px;' +
    'color:#1c5cab;font-weight:600;' + PANEL;
  document.body.appendChild(daiBtn);

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
    'padding:10px 16px calc(16px + env(safe-area-inset-bottom));' +
    'font:12.5px/1.6 -apple-system,sans-serif;color:' + INK;
  drawer.innerHTML =
    '<div style="width:36px;height:4px;border-radius:2px;background:#d5d4cc;margin:0 auto 10px"></div>' +
    '<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">' +
    '<b style="font-size:14px">大区分</b>' +
    '<span style="color:' + MUTED + ';font-size:11.5px">タップで表示切替</span></div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px">' +
    anchors.map(function (t, i) {
      return '<div class="ka-dai-item" data-i="' + i + '" style="display:flex;align-items:center;' +
        'min-width:0;gap:7px;border:1px solid ' + LINE + ';border-radius:9px;padding:7px 9px;' +
        'background:#fff;transition:opacity .15s;' + (t.vis ? '' : 'opacity:0.35') + '">' +
        '<span style="flex:none;width:11px;height:11px;border-radius:50%;background:' + t.color + '"></span>' +
        '<span style="flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' +
        esc(t.label) + '</span>' +
        '<span style="color:' + MUTED + ';font-size:11px">' + fmt(t.n) + '</span></div>';
    }).join('') + '</div>';
  document.body.appendChild(drawer);

  function openDrawer(on) {
    drawer.style.transform = on ? 'translateY(0)' : 'translateY(105%)';
    drawer.dataset.open = on ? '1' : '';
    backdrop.style.opacity = on ? '1' : '0';
    backdrop.style.visibility = on ? 'visible' : 'hidden';
    daiBtn.style.display = on ? 'none' : '';
  }
  daiBtn.addEventListener('click', function () { openDrawer(true); });
  backdrop.addEventListener('click', function () { openDrawer(false); });
  drawer.querySelectorAll('.ka-dai-item').forEach(function (el) {
    el.addEventListener('click', function () {
      var t = anchors[parseInt(el.getAttribute('data-i'), 10)];
      daiOn[t.dai] = !daiOn[t.dai];
      el.style.opacity = daiOn[t.dai] ? '' : '0.35';
      setDai(t.dai, daiOn[t.dai]);
    });
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

// ---- 3Dの2本指操作（タッチ端末のみ。gl3dはピンチが効かない環境があるため自前実装）:
// 指の間隔の変化 = 拡大縮小（視点距離をスケール）、2本指の重心の移動 = 並行移動
// （視点と注視点を画面平面に沿って同じだけ動かす）。両方を同時に扱う。1本指回転はPlotly標準 ----
function v3(x, y, z) { return { x: x, y: y, z: z }; }
function vsub(a, b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
function vadd(a, b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
function vmul(a, k) { return v3(a.x * k, a.y * k, a.z * k); }
function vcross(a, b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
function vlen(a) { return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
function vnorm(a) { var l = vlen(a) || 1; return vmul(a, 1 / l); }
function tCenter(t) { return [(t[0].clientX + t[1].clientX) / 2, (t[0].clientY + t[1].clientY) / 2]; }
if (isTouch && is3d) {
  var pinch3 = null, raf3 = false;
  plot.addEventListener('touchstart', function (e) {
    if (e.touches.length !== 2) return;
    e.stopPropagation();
    var cam = (plot._fullLayout.scene && plot._fullLayout.scene.camera) || {};
    var eye = cam.eye || v3(1.25, 1.25, 1.25);
    var ctr = cam.center || v3(0, 0, 0);
    var up = cam.up || v3(0, 0, 1);
    // カメラ基底: 視線 f、画面右 r、画面上 u
    var f = vnorm(vsub(ctr, eye));
    var r = vnorm(vcross(f, up));
    var u = vcross(r, f);
    var c0 = tCenter(e.touches);
    pinch3 = { d0: tDist(e.touches), cx: c0[0], cy: c0[1],
               eye: v3(eye.x, eye.y, eye.z), ctr: v3(ctr.x, ctr.y, ctr.z), r: r, u: u,
               // 1ピクセルあたりの空間距離（透視投影 fovy=45° で注視点距離の画面高さから換算）
               k: 2 * vlen(vsub(eye, ctr)) * Math.tan(Math.PI / 8) / plot._fullLayout._size.h };
  }, { capture: true, passive: true });
  plot.addEventListener('touchmove', function (e) {
    if (!pinch3 || e.touches.length !== 2) return;
    e.preventDefault(); e.stopPropagation();
    var s = Math.max(0.05, pinch3.d0 / tDist(e.touches));  // 指を広げる=s<1=近づく
    var c = tCenter(e.touches), mx = c[0] - pinch3.cx, my = c[1] - pinch3.cy;
    if (raf3) return;
    raf3 = true;
    requestAnimationFrame(function () {
      raf3 = false;
      if (!pinch3) return;
      var p = pinch3;
      // 指を右へ動かす=場面が右へ=カメラは左へ（-r）。画面yは下向きなので下へ動かす=カメラは上へ（+u）
      var T = vadd(vmul(p.r, -mx * p.k), vmul(p.u, my * p.k));
      Plotly.relayout(plot, {
        'scene.camera.eye': vadd(vadd(p.ctr, vmul(vsub(p.eye, p.ctr), s)), T),
        'scene.camera.center': vadd(p.ctr, T),
      });
    });
  }, { capture: true, passive: false });
  plot.addEventListener('touchend', function (e) {
    if (e.touches.length < 2) pinch3 = null;
  }, { capture: true, passive: true });
}

}  // setupUI
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
