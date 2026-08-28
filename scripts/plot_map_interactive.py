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

// ---- 自前ツールチップ（固定サイズ・常にカーソル右側） ----
var tip = document.createElement('div');
tip.style.cssText = 'position:fixed;display:none;z-index:1000;background:#fff;' +
  'border:1.5px solid #999;border-radius:4px;padding:6px 9px;' +
  'font:12px/1.5 sans-serif;color:#0b0b0b;pointer-events:none;width:320px;' +
  'box-shadow:0 2px 8px rgba(0,0,0,0.15)';
document.body.appendChild(tip);
var mx = 0, my = 0, hovered = null, lastK = null, lastT = 0, suppressUntil = 0;
function place() { tip.style.left = (mx + 16) + 'px'; tip.style.top = (my + 12) + 'px'; }
document.addEventListener('mousemove', function (e) {
  mx = e.clientX; my = e.clientY;
  if (tip.style.display !== 'none') place();
});
plot.on('plotly_hover', function (d) {
  var p = d.points[0];
  hovered = (p && p.customdata) || null;
  if (!p || !p.text) return;
  var color = (p.fullData && p.fullData.marker && p.fullData.marker.color) || '#999';
  var lines = p.text.split('<br>');
  var ell = 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
  tip.innerHTML =
    '<div style="' + ell + ';background:' + color + ';color:#fff;font-weight:600;' +
    'margin:-6px -9px 4px -9px;padding:4px 9px;border-radius:2.5px 2.5px 0 0">' +
    lines[0] + '</div>' +
    lines.slice(1).map(function (x) {
      return '<div style="' + ell + '">' + x + '</div>';
    }).join('');
  tip.style.borderColor = color;
  tip.style.display = 'block'; place();
});
plot.on('plotly_unhover', function () { hovered = null; tip.style.display = 'none'; });

// ---- ホバー中の点をクリックで KAKEN ページ ----
plot.on('plotly_doubleclick', function () { suppressUntil = Date.now() + 700; });
plot.on('plotly_relayout', function () { suppressUntil = Date.now() + 400; });
plot.on('plotly_click', function (d) {
  var k = d.points[0] && d.points[0].customdata;
  var now = Date.now();
  if (!k || now < suppressUntil) return;
  if (k !== hovered) return;
  if (k === lastK && now - lastT < 600) return;
  lastK = k; lastT = now;
  window.open('https://kaken.nii.ac.jp/ja/grant/' + k + '/', '_blank');
});

// ---- 種目フィルタ（トレース＝大区分×種目。表示/非表示切替のみで高速） ----
// チェック解除 → visible=false。チェック → 同じ大区分グループ内で種目オフでない
// トレースの表示状態（true / legendonly）を継承し、大区分の凡例スイッチと両立させる。
var counts = {};
plot.data.forEach(function (t) {
  if (t.meta) counts[t.meta] = (counts[t.meta] || 0) + (t.customdata ? t.customdata.length : 0);
});
var order = __CAT_ORDER__;
var cats = order.filter(function (c) { return counts[c] !== undefined; }).concat(
  Object.keys(counts).filter(function (c) { return order.indexOf(c) < 0; })
    .sort(function (a, b) { return counts[b] - counts[a]; }));

var panel = document.createElement('div');
panel.style.cssText = 'position:fixed;top:56px;left:10px;z-index:999;' +
  'background:rgba(252,252,251,0.96);border:1px solid #ccc;border-radius:6px;' +
  'padding:8px 12px;font:12px/1.7 sans-serif;color:#0b0b0b;max-height:70vh;' +
  'overflow-y:auto;box-shadow:0 2px 8px rgba(0,0,0,0.12)';
panel.innerHTML = '<b>種目フィルタ</b><span id="ka-hint" style="color:#898781"> ▸</span>' +
  '<div id="ka-body" style="display:none">' +
  '<a href="#" id="ka-selall">全選択</a>&nbsp;<a href="#" id="ka-selnone">全解除</a><br>' +
  cats.map(function (c) {
    return '<label style="display:block;white-space:nowrap">' +
      '<input type="checkbox" class="ka-cat" checked> ' +
      c + ' (' + counts[c].toLocaleString() + ')</label>';
  }).join('') + '</div>';
document.body.appendChild(panel);
// 普段は畳んでおき、ホバーで展開する
var body = document.getElementById('ka-body');
var hint = document.getElementById('ka-hint');
panel.addEventListener('mouseenter', function () { body.style.display = 'block'; hint.textContent = ''; });
panel.addEventListener('mouseleave', function () { body.style.display = 'none'; hint.textContent = ' ▸'; });
var boxes = panel.querySelectorAll('.ka-cat');

function setCategory(cat, on) {
  var idx = [], vis = [];
  plot.data.forEach(function (t, i) {
    if (t.meta !== cat) return;
    if (!on) { idx.push(i); vis.push(false); return; }
    var v = true;  // 大区分グループの現在の表示状態を継承
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
"""


def main() -> None:
    coords_path = Path(sys.argv[1])
    coords = pl.read_parquet(coords_path)
    is_3d = "c2" in coords.columns
    corpus = pl.read_parquet(
        "data/processed/corpus.parquet", columns=["award_number", "kaken_id", "category"]
    )
    titles = pl.read_parquet(  # 英語タイトル補完済みの awards から
        "data/interim/awards.parquet", columns=["award_number", "title"]
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
                customdata=sub["kaken_id"].to_list(),
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

    dims = "3D（ドラッグで回転）" if is_3d else "2D（スクロールで拡大縮小・ドラッグで移動・ダブルクリックで全体表示）"
    layout = dict(
        title=f"科研費 学術地図 {dims} / 凡例クリックで区分の表示切替 / 点をクリックでKAKENページ",
        paper_bgcolor=SURFACE,
        legend=dict(itemsizing="constant", font=dict(size=11), groupclick="togglegroup"),
        annotations=[dict(text=FOOTER, x=0, y=0, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=9, color="#898781"))],
        margin=dict(l=0, r=0, t=50, b=30),
    )
    if is_3d:
        layout["scene"] = dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor=SURFACE,
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
        post_script=POST_SCRIPT.replace("__CAT_ORDER__", json.dumps(CATEGORY_ORDER, ensure_ascii=False)),
    )
    n_traces = len(fig.data)
    print(f"出力: {out} ({out.stat().st_size / 1e6:.1f} MB, {n_traces} トレース)")


if __name__ == "__main__":
    main()
