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
        cat_counts = dsub.group_by("category").len().sort("len", descending=True)
        for cat in cat_counts["category"]:
            sub = dsub.filter(pl.col("category") == cat)
            traces.append(dict(
                k="d", dai=dai, label=label, color=color, cat=cat,
                off=offset, n=sub.height, vis=visible,
            ))
            parts.append(sub)
            offset += sub.height
    return pl.concat(parts), traces


def main() -> None:
    coords_path = Path(sys.argv[1])
    coords = pl.read_parquet(coords_path)
    is_3d = "c2" in coords.columns
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

    # 座標の量子化: 各軸を int16 全域に線形写像（分解能=値域/65535、1ピクセル未満）
    quant = []
    qarrs = []
    for c in dims:
        v = big[c].to_numpy()
        lo, hi = float(v.min()), float(v.max())
        qarrs.append(np.round((v - lo) / (hi - lo) * 65535 - 32768).astype("<i2"))
        quant.append(dict(lo=lo, hi=hi))
    points_bin = b"".join(a.tobytes() for a in qarrs)

    out_dir = Path(f"docs/map{'3d' if is_3d else '2d'}")
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
        n=n, is3d=is_3d, shardSize=SHARD_SIZE, nShards=n_shards,
        pointsBytes=len(points_bin), quant=quant, ktypes=ktypes,
        traces=traces, catOrder=CATEGORY_ORDER, footer=FOOTER,
        title=f"科研費 学術地図 {'3D' if is_3d else '2D'}",
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
<script src="__PLOTLY_CDN__" charset="utf-8"></script>
<style>
  body { margin:0; background:#fcfcfb; }
  #plot { margin-top:48px; height:calc(100vh - 48px); }
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
var SHARD_SHIFT = Math.log2(M.shardSize);

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
      marker: is3d ? { size: 1.3, color: t.color, opacity: 0.55 }
                   : { size: 2.2, color: t.color, opacity: 0.5 },
      type: is3d ? 'scatter3d' : 'scattergl',
    };
    if (is3d) d.z = zs.subarray(t.off, end);
    data.push(d); gidOffset.push(t.off);
  });

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
  } else {
    layout.plot_bgcolor = SURFACE;
    layout.xaxis = { visible: false };
    layout.yaxis = { visible: false, scaleanchor: 'x' };
    layout.dragmode = 'pan';
  }
  await Plotly.newPlot(plot, data, layout,
    { scrollZoom: true, displaylogo: false, doubleClick: 'reset', responsive: true });

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
bar.style.cssText = 'position:fixed;top:0;left:0;right:0;height:48px;z-index:998;' +
  'display:flex;align-items:center;gap:14px;padding:0 18px;background:#fcfcfb;' +
  'border-bottom:1px solid ' + LINE + ';font:13px -apple-system,sans-serif;color:' + INK;
bar.innerHTML =
  '<div style="white-space:nowrap"><b style="font-size:14.5px">' + M.title + '</b>' +
  ' <span style="color:' + MUTED + ';font-size:12px">' + M.sub + '</span></div>' +
  '<div style="position:relative;flex:0 1 300px;min-width:170px">' +
  '  <input id="ka-q" type="search" placeholder="詳細データ読み込み中…" disabled' +
  '   style="width:100%;box-sizing:border-box;padding:6px 12px;border:1px solid #cfcec7;' +
  '   border-radius:15px;background:#fff;font:12.5px -apple-system,sans-serif;outline:none">' +
  '  <div id="ka-results" style="display:none;position:absolute;top:36px;left:0;width:380px;' +
  '   max-height:55vh;overflow-y:auto;padding:8px 12px;' + PANEL + '"></div>' +
  '</div>' +
  '<div style="flex:1"></div>' +
  '<div id="ka-filter-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span style="cursor:default;color:' + SUB + '">種目フィルタ ▾</span>' +
  '  <div id="ka-body" style="display:none;position:absolute;top:100%;right:0;' +
  '   max-height:70vh;overflow-y:auto;padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>' +
  '<div id="ka-help-wrap" style="position:relative;align-self:stretch;display:flex;align-items:center">' +
  '  <span style="cursor:default;color:' + SUB + '">操作 ▾</span>' +
  '  <div id="ka-help-body" style="display:none;position:absolute;top:100%;right:0;' +
  '   padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>';
document.body.insertBefore(bar, document.body.firstChild);
if (allLoaded) finishPrefetch();  // 先読みがヘッダー生成より先に終わった場合の取りこぼし

// ホバーで開閉。ボタン→パネルへの斜め移動で一瞬外に出ても閉じないよう、350msの閉じ猶予
[['ka-filter-wrap', 'ka-body'], ['ka-help-wrap', 'ka-help-body']].forEach(function (pair) {
  var wrap = document.getElementById(pair[0]);
  var body = document.getElementById(pair[1]);
  var closeTimer = null;
  wrap.addEventListener('mouseenter', function () {
    if (closeTimer) { clearTimeout(closeTimer); closeTimer = null; }
    body.style.display = 'block';
  });
  wrap.addEventListener('mouseleave', function () {
    closeTimer = setTimeout(function () { body.style.display = 'none'; }, 350);
  });
});

document.getElementById('ka-help-body').innerHTML = is3d
  ? '<div>ドラッグ: 回転 / スクロール: 拡大縮小</div>' +
    '<div>点にホバー→クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替</div>'
  : '<div>スクロール: 拡大縮小 / ドラッグ: 移動</div>' +
    '<div>ダブルクリック: 全体表示に戻る</div>' +
    '<div>点にホバー→クリック: KAKENページを開く</div>' +
    '<div>凡例クリック: 大区分の表示切替</div>' +
    '<div>ツールバーのなげなわ/矩形: 囲って集計</div>' +
    '<div>Esc: 選択解除</div>';

// ---- 自前ツールチップ（固定サイズ・常にカーソル右側） ----
var tip = document.createElement('div');
tip.style.cssText = 'position:fixed;display:none;z-index:1000;background:#fff;' +
  'border:1.5px solid #999;border-radius:6px;padding:6px 9px;' +
  'font:12px/1.5 -apple-system,sans-serif;color:' + INK + ';pointer-events:none;width:320px;' +
  'box-shadow:0 2px 8px rgba(0,0,0,0.15)';
document.body.appendChild(tip);
var mx = 0, my = 0, hoveredGid = null, lastK = null, lastT = 0, suppressUntil = 0;
function place() { tip.style.left = (mx + 16) + 'px'; tip.style.top = (my + 12) + 'px'; }
document.addEventListener('mousemove', function (e) {
  mx = e.clientX; my = e.clientY;
  if (tip.style.display !== 'none') place();
});
function gidOf(p) {
  if (!p || gidOffset[p.curveNumber] < 0) return null;
  return gidOffset[p.curveNumber] + p.pointNumber;
}
function renderTip(gid, tr) {
  var row = getRow(gid);
  var ell = 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
  var title = row ? esc((row[2] || '（タイトルなし）').slice(0, 48))
                  : '<span style="color:' + MUTED + '">（読み込み中…）</span>';
  var tail = row ? esc(tr.cat + ' / ' + row[0]) : esc(tr.cat);
  tip.innerHTML =
    '<div style="' + ell + ';background:' + tr.color + ';color:#fff;font-weight:600;' +
    'margin:-6px -9px 4px -9px;padding:4px 9px;border-radius:4.5px 4.5px 0 0">' +
    esc(tr.label) + '</div>' +
    '<div style="' + ell + '">' + title + '</div>' +
    '<div style="' + ell + '">' + tail + '</div>';
  tip.style.borderColor = tr.color;
  tip.style.display = 'block'; place();
}
plot.on('plotly_hover', function (d) {
  var p = d.points[0];
  var gid = gidOf(p);
  if (gid === null) return;
  hoveredGid = gid;
  var tr = M.traces[traceOf[gid]];
  renderTip(gid, tr);
  if (!getRow(gid)) {  // 未取得シャードはその場で取得し、まだ同じ点なら描き直す
    ensureShard(gid >> SHARD_SHIFT).then(function () {
      if (hoveredGid === gid) renderTip(gid, tr);
    }).catch(function () {});
  }
});
plot.on('plotly_unhover', function () { hoveredGid = null; tip.style.display = 'none'; });

// ---- ホバー中の点をクリックで KAKEN ページ ----
plot.on('plotly_doubleclick', function () { suppressUntil = Date.now() + 700; });
plot.on('plotly_relayout', function () { suppressUntil = Date.now() + 400; });
plot.on('plotly_click', function (d) {
  var gid = gidOf(d.points[0]);
  var now = Date.now();
  if (gid === null || now < suppressUntil) return;
  if (gid !== hoveredGid) return;
  var row = getRow(gid);
  if (!row) return;  // ホバー時に取得が走っているので、直後の再クリックで開ける
  var k = kakenId(row);
  if (k === lastK && now - lastT < 600) return;
  lastK = k; lastT = now;
  window.open('https://kaken.nii.ac.jp/ja/grant/' + k + '/', '_blank');
});

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
      if (is3d) return;
      var span = 1.5;
      Plotly.relayout(plot, {
        'xaxis.range': [xs[h.gid] - span, xs[h.gid] + span],
        'yaxis.range': [ys[h.gid] - span, ys[h.gid] + span],
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
  Plotly.update(plot, { selectedpoints: null }, { selections: [], dragmode: 'pan' });
});

}  // setupUI
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
