"""UMAP 座標（2D/3D）からインタラクティブ地図（自己完結 HTML）を生成する。

- 座標 parquet の列（c0,c1[,c2]）から 2D / 3D を自動判別
- トレース＝大区分×種目。凡例は大区分単位（legendgroup）、
  種目フィルタはトレースの表示/非表示切替（データ転送なしで高速）
- 自前ツールチップ（固定サイズ・常にカーソル右側）: 大区分 / タイトル / 種目・課題番号
- ホバー表示中の点をクリックで KAKEN の課題ページを新しいタブに開く
- 配色は意味順色相環（kaken_atlas.kubun.DAI_COLORS）

使い方:
    uv run python scripts/plot_map_interactive.py data/processed/umap2d_nn15_md0.1.parquet
    uv run python scripts/plot_map_interactive.py data/processed/umap3d_nn15_md0.1.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, "src")
from kaken_atlas.kubun import DAI_COLORS, DAI_GLOSS, load_dai_labels  # noqa: E402

SURFACE = "#fcfcfb"

# 種目フィルタの表示順（ユーザ指定、2026-08-28。ここにない種目は末尾に件数順で並ぶ）
CATEGORY_ORDER = [
    "特別推進研究",
    "基盤研究(S)",
    "基盤研究(A)",
    "基盤研究(B)",
    "基盤研究(C)",
    "挑戦的研究(開拓)",
    "挑戦的研究(萌芽)",
    "若手研究",
    "若手研究(B)",
    "研究活動スタート支援",
    "特別研究員奨励費",
    "新学術領域研究(研究領域提案型)",
    "学術変革領域研究(A)",
    "学術変革領域研究(B)",
    "学術変革領域研究(学術研究支援基盤形成)",
    "国際共同研究加速基金(国際先導研究)",
    "国際共同研究加速基金(国際共同研究強化)",
    "国際共同研究加速基金(国際共同研究強化(A))",
    "国際共同研究加速基金(国際共同研究強化(B))",
    "国際共同研究加速基金(海外連携研究)",
    "国際共同研究加速基金(帰国発展研究)",
    "奨励研究",
    "特別研究促進費",
]

FOOTER = (
    "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | "
    "2019–2025年度開始の採択課題 206,078件（採択時概要あり・不採択除く） | "
    "埋め込み: cl-nagoya/ruri-v3-310m | UMAP (cosine, n_neighbors=15, seed=42) | "
    "作成: KAKEN-ATLAS (26K15524)"
)

# 自前ツールチップ・クリックで KAKEN ページ・種目フィルタ。
# customdata = kaken_id（点ごと）、meta = 種目名（トレースごと）。
POST_SCRIPT = """
var plot = document.getElementsByClassName('plotly-graph-div')[0];
var is3d = plot.data.length && plot.data[0].type === 'scatter3d';

// ==== デザイントークン ====
var INK = '#0b0b0b', MUTED = '#898781', SUB = '#52514e', LINE = '#e1e0d9';
var PANEL = 'background:#fcfcfb;border:1px solid ' + LINE + ';border-radius:8px;' +
  'box-shadow:0 3px 12px rgba(0,0,0,0.10);font:12.5px/1.7 -apple-system,sans-serif;color:' + INK;

// ==== ヘッダーバー ====
document.body.style.margin = '0';
var bar = document.createElement('div');
bar.style.cssText = 'position:fixed;top:0;left:0;right:0;height:48px;z-index:998;' +
  'display:flex;align-items:center;gap:14px;padding:0 18px;background:#fcfcfb;' +
  'border-bottom:1px solid ' + LINE + ';font:13px -apple-system,sans-serif;color:' + INK;
bar.innerHTML =
  '<div style="white-space:nowrap"><b style="font-size:14.5px">__MAP_TITLE__</b>' +
  ' <span style="color:' + MUTED + ';font-size:12px">__MAP_SUB__</span></div>' +
  '<div style="position:relative;flex:0 1 300px;min-width:170px">' +
  '  <input id="ka-q" type="search" placeholder="タイトル・キーワード・課題番号を検索"' +
  '   style="width:100%;box-sizing:border-box;padding:6px 12px;border:1px solid #cfcec7;' +
  '   border-radius:15px;background:#fff;font:12.5px -apple-system,sans-serif;outline:none">' +
  '  <div id="ka-results" style="display:none;position:absolute;top:36px;left:0;width:380px;' +
  '   max-height:55vh;overflow-y:auto;padding:8px 12px;' + PANEL + '"></div>' +
  '</div>' +
  '<div style="flex:1"></div>' +
  '<div id="ka-filter-wrap" style="position:relative">' +
  '  <span style="cursor:default;color:' + SUB + '">種目フィルタ ▾</span>' +
  '  <div id="ka-body" style="display:none;position:absolute;top:26px;right:0;' +
  '   max-height:70vh;overflow-y:auto;padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>' +
  '<div id="ka-help-wrap" style="position:relative">' +
  '  <span style="cursor:default;color:' + SUB + '">操作 ▾</span>' +
  '  <div id="ka-help-body" style="display:none;position:absolute;top:26px;right:0;' +
  '   padding:8px 14px;white-space:nowrap;' + PANEL + '"></div>' +
  '</div>';
document.body.insertBefore(bar, document.body.firstChild);

// 地図をヘッダー分下げる
plot.style.marginTop = '48px';
plot.style.height = 'calc(100vh - 48px)';
if (window.Plotly && Plotly.Plots) Plotly.Plots.resize(plot);

// ホバーで開閉（ボタンとパネルを同じラッパに入れてあるので間の移動で閉じない）
[['ka-filter-wrap', 'ka-body'], ['ka-help-wrap', 'ka-help-body']].forEach(function (pair) {
  var wrap = document.getElementById(pair[0]);
  var body = document.getElementById(pair[1]);
  wrap.addEventListener('mouseenter', function () { body.style.display = 'block'; });
  wrap.addEventListener('mouseleave', function () { body.style.display = 'none'; });
});

// 操作ヘルプの中身
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

// ==== 自前ツールチップ（固定サイズ・常にカーソル右側） ====
var tip = document.createElement('div');
tip.style.cssText = 'position:fixed;display:none;z-index:1000;background:#fff;' +
  'border:1.5px solid #999;border-radius:6px;padding:6px 9px;' +
  'font:12px/1.5 -apple-system,sans-serif;color:' + INK + ';pointer-events:none;width:320px;' +
  'box-shadow:0 2px 8px rgba(0,0,0,0.15)';
document.body.appendChild(tip);
var mx = 0, my = 0, hovered = null, lastK = null, lastT = 0, suppressUntil = 0;
function place() { tip.style.left = (mx + 16) + 'px'; tip.style.top = (my + 12) + 'px'; }
document.addEventListener('mousemove', function (e) {
  mx = e.clientX; my = e.clientY;
  if (tip.style.display !== 'none') place();
});
function kakenId(p) {
  var cd = p && p.customdata;
  return Array.isArray(cd) ? cd[0] : cd;
}
plot.on('plotly_hover', function (d) {
  var p = d.points[0];
  hovered = kakenId(p);
  if (!p || !p.text) return;
  var color = (p.fullData && p.fullData.marker && p.fullData.marker.color) || '#999';
  var lines = p.text.split('<br>');
  var ell = 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
  tip.innerHTML =
    '<div style="' + ell + ';background:' + color + ';color:#fff;font-weight:600;' +
    'margin:-6px -9px 4px -9px;padding:4px 9px;border-radius:4.5px 4.5px 0 0">' +
    lines[0] + '</div>' +
    lines.slice(1).map(function (x) {
      return '<div style="' + ell + '">' + x + '</div>';
    }).join('');
  tip.style.borderColor = color;
  tip.style.display = 'block'; place();
});
plot.on('plotly_unhover', function () { hovered = null; tip.style.display = 'none'; });

// ==== ホバー中の点をクリックで KAKEN ページ ====
plot.on('plotly_doubleclick', function () { suppressUntil = Date.now() + 700; });
plot.on('plotly_relayout', function () { suppressUntil = Date.now() + 400; });
plot.on('plotly_click', function (d) {
  var k = kakenId(d.points[0]);
  var now = Date.now();
  if (!k || now < suppressUntil) return;
  if (k !== hovered) return;
  if (k === lastK && now - lastT < 600) return;
  lastK = k; lastT = now;
  window.open('https://kaken.nii.ac.jp/ja/grant/' + k + '/', '_blank');
});

// ==== 検索（タイトル・キーワード・課題番号の部分一致 → ハイライト＋一覧） ====
var qInput = document.getElementById('ka-q');
var qResults = document.getElementById('ka-results');
var hlIndex = null, qTimer = null;

function clearHighlight() {
  if (hlIndex !== null) { Plotly.deleteTraces(plot, hlIndex); hlIndex = null; }
  qResults.style.display = 'none';
}
function runSearch(q) {
  clearHighlight();
  q = q.trim().toLowerCase();
  if (q.length < 2) return;
  var hits = [];
  for (var i = 0; i < plot.data.length; i++) {
    var t = plot.data[i];
    if (!t.meta || !t.text) continue;
    if (t.visible === false || t.visible === 'legendonly') continue;
    var fd = plot._fullData[i];
    for (var j = 0; j < t.text.length; j++) {
      var cd = t.customdata[j];
      var hay = (t.text[j] + '\u3001' + (cd && cd[1] || '')).toLowerCase();
      if (hay.indexOf(q) < 0) continue;
      var h = { x: fd.x[j], y: fd.y[j], text: t.text[j], kaken: cd && cd[0] };
      if (is3d) h.z = fd.z[j];
      hits.push(h);
      if (hits.length >= 2000) break;
    }
    if (hits.length >= 2000) break;
  }
  if (!hits.length) {
    qResults.innerHTML = '<span style="color:' + MUTED + '">該当なし</span>';
    qResults.style.display = 'block';
    return;
  }
  var overlay = {
    x: hits.map(function (h) { return h.x; }),
    y: hits.map(function (h) { return h.y; }),
    mode: 'markers', hoverinfo: 'none', showlegend: false,
    marker: { size: is3d ? 4 : 9, color: 'rgba(11,11,11,0)',
              line: { width: 2, color: INK } },
    type: is3d ? 'scatter3d' : 'scattergl',
  };
  if (is3d) overlay.z = hits.map(function (h) { return h.z; });
  Plotly.addTraces(plot, overlay).then(function () { hlIndex = plot.data.length - 1; });
  var html = '<b>' + hits.length.toLocaleString() + (hits.length >= 2000 ? '+' : '') +
    '件ヒット</b><span style="color:' + MUTED + '">（先頭30件）</span><br>';
  hits.slice(0, 30).forEach(function (h, k) {
    var lines = h.text.split('<br>');
    html += '<a href="#" class="ka-hit" data-k="' + k + '" style="display:block;' +
      'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#1c5cab;' +
      'text-decoration:none;padding:1px 0">' + lines[1] + ' <span style="color:' + MUTED + '">' +
      lines[2] + '</span></a>';
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
        'xaxis.range': [h.x - span, h.x + span],
        'yaxis.range': [h.y - span, h.y + span],
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

// ==== 種目フィルタ ====
var counts = {};
plot.data.forEach(function (t) {
  if (t.meta) counts[t.meta] = (counts[t.meta] || 0) + (t.customdata ? t.customdata.length : 0);
});
var order = __CAT_ORDER__;
var cats = order.filter(function (c) { return counts[c] !== undefined; }).concat(
  Object.keys(counts).filter(function (c) { return order.indexOf(c) < 0; })
    .sort(function (a, b) { return counts[b] - counts[a]; }));

document.getElementById('ka-body').innerHTML =
  '<a href="#" id="ka-selall" style="color:#1c5cab;text-decoration:none">全選択</a>&nbsp; ' +
  '<a href="#" id="ka-selnone" style="color:#1c5cab;text-decoration:none">全解除</a>' +
  cats.map(function (c) {
    return '<label style="display:block;white-space:nowrap;cursor:pointer">' +
      '<input type="checkbox" class="ka-cat" checked style="vertical-align:-2px"> ' +
      c + ' <span style="color:' + MUTED + '">' + counts[c].toLocaleString() + '</span></label>';
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

// ==== 選択パネル（なげなわ/矩形で囲うと内訳・キーワード集計を即時表示） ====
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
  var n = d.points.length, dais = {}, cts = {}, kws = {};
  d.points.forEach(function (p) {
    var g = p.fullData.legendgroup || '?';
    dais[g] = (dais[g] || 0) + 1;
    var m = p.fullData.meta || '?';
    cts[m] = (cts[m] || 0) + 1;
    var cd = p.customdata;
    if (Array.isArray(cd) && cd[1]) {
      cd[1].split('、').forEach(function (w) { if (w) kws[w] = (kws[w] || 0) + 1; });
    }
  });
  var html = '<b>選択: ' + n.toLocaleString() + '件</b>' +
    ' <a href="#" id="ka-selclear" style="color:' + MUTED + '">閉じる</a><br>';
  html += '<span style="color:' + SUB + '">大区分:</span> ' + topEntries(dais, 5).map(function (g) {
    return g + ' ' + dais[g].toLocaleString();
  }).join(' / ') + '<br>';
  html += '<span style="color:' + SUB + '">種目:</span> ' + topEntries(cts, 4).map(function (c) {
    return c + ' ' + cts[c].toLocaleString();
  }).join(' / ') + '<br>';
  html += '<span style="color:' + SUB + '">頻出キーワード:</span><br>' +
    topEntries(kws, 15).map(function (w) {
      return '<span style="display:inline-block;background:#eef3fa;border:1px solid #c9d8ee;' +
        'border-radius:4px;padding:0 6px;margin:1px 2px">' + w +
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
"""


def main() -> None:
    coords_path = Path(sys.argv[1])
    coords = pl.read_parquet(coords_path)
    is_3d = "c2" in coords.columns
    corpus = pl.read_parquet(
        "data/processed/corpus.parquet", columns=["award_number", "kaken_id", "category"]
    )
    titles = pl.read_parquet(  # 英語タイトル補完済み。keywords は選択パネルの集計用
        "data/interim/awards.parquet", columns=["award_number", "title", "keywords"]
    )
    df = coords.join(corpus, on="award_number", how="left")
    df = df.join(titles, on="award_number", how="left")
    df = df.join(load_dai_labels(), on="award_number", how="left")

    fig = go.Figure()
    for dai in [*DAI_COLORS.keys(), "複数", "区分なし"]:
        dsub = df.filter(pl.col("dai") == dai)
        color = DAI_COLORS.get(dai, "#b9b8b0")
        label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
        # 凡例アンカー: 空の点1つだけのトレースが大区分の凡例見出しを担う。
        # データトレースは種目フィルタで消え得るため、凡例は常在のアンカーに背負わせる。
        anchor = dict(
            mode="markers",
            name=f"{label} {dsub.height:,}",
            legendgroup=dai,
            showlegend=True,
            hoverinfo="none",
            visible=True if dai != "区分なし" else "legendonly",
            marker=dict(size=6, color=color, opacity=0.9),
        )
        if is_3d:
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], **anchor))
        else:
            fig.add_trace(go.Scattergl(x=[None], y=[None], **anchor))
        # 大区分の中を種目別トレースに分割
        cat_counts = dsub.group_by("category").len().sort("len", descending=True)
        for cat in cat_counts["category"]:
            sub = dsub.filter(pl.col("category") == cat)
            text = [
                f"{label}<br>{(t or '（タイトルなし）')[:48]}<br>{cat} / {a}"
                for t, a in zip(sub["title"], sub["award_number"], strict=True)
            ]
            common = dict(
                mode="markers",
                name=f"{label} {dsub.height:,}",
                meta=cat,  # 種目フィルタがトレースを種目に対応付けるのに使う
                legendgroup=dai,
                showlegend=False,  # 凡例はアンカーが担う
                text=text,
                hoverinfo="none",  # 吹き出しは自前ツールチップ（POST_SCRIPT）で描く
                customdata=[
                    [k, "、".join(kw)] for k, kw in
                    zip(sub["kaken_id"], sub["keywords"], strict=True)
                ],
                visible=True if dai != "区分なし" else "legendonly",
            )
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=sub["c0"], y=sub["c1"], z=sub["c2"],
                    marker=dict(size=1.3, color=color, opacity=0.55), **common,
                ))
            else:
                fig.add_trace(go.Scattergl(
                    x=sub["c0"], y=sub["c1"],
                    marker=dict(size=2.2, color=color, opacity=0.5), **common,
                ))

    layout = dict(
        paper_bgcolor=SURFACE,
        legend=dict(itemsizing="constant", font=dict(size=11), groupclick="togglegroup"),
        annotations=[dict(text=FOOTER, x=0, y=0, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=9, color="#898781"))],
        margin=dict(l=0, r=0, t=24, b=30),
    )
    if is_3d:
        layout["scene"] = dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor=SURFACE,
            dragmode="orbit",  # つっかかりのない自由回転（定量分析には使わない前提）
        )
    else:
        layout.update(
            plot_bgcolor=SURFACE,
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
            dragmode="pan",
        )
    fig.update_layout(**layout)

    out = Path(f"reports/figures/map_{'3d' if is_3d else '2d'}_interactive.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        out, include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"},
        post_script=(
            POST_SCRIPT
            .replace("__CAT_ORDER__", json.dumps(CATEGORY_ORDER, ensure_ascii=False))
            .replace("__MAP_TITLE__", f"科研費 学術地図 {'3D' if is_3d else '2D'}")
            .replace("__MAP_SUB__", "2019–2025年度・206,078件")
        ),
    )
    n_traces = len(fig.data)
    print(f"出力: {out} ({out.stat().st_size / 1e6:.1f} MB, {n_traces} トレース)")


if __name__ == "__main__":
    main()
