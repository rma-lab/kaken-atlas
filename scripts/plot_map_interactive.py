"""UMAP 座標（2D/3D）からインタラクティブ地図（自己完結 HTML）を生成する。

- 座標 parquet の列（c0,c1[,c2]）から 2D / 3D を自動判別
- 大区分ごとに1トレース（凡例クリックで表示/非表示、ダブルクリックで単独表示）
- ホバーで課題タイトル・種目・課題番号
- 配色は意味順色相環（kaken_atlas.kubun.DAI_COLORS）

使い方:
    uv run python scripts/plot_map_interactive.py data/processed/umap2d_nn15_md0.1.parquet
    uv run python scripts/plot_map_interactive.py data/processed/umap3d_nn15_md0.1.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

sys.path.insert(0, "src")
from kaken_atlas.kubun import DAI_COLORS, DAI_GLOSS, load_dai_labels  # noqa: E402

SURFACE = "#fcfcfb"
FOOTER = (
    "データ: KAKEN科研費データベース（国立情報学研究所）より取得 | "
    "2019–2025年度開始の採択課題 206,078件（採択時概要あり・不採択除く） | "
    "埋め込み: cl-nagoya/ruri-v3-310m | UMAP (cosine, n_neighbors=15, seed=42) | "
    "作成: KAKEN-ATLAS (26K15524)"
)


def main() -> None:
    coords_path = Path(sys.argv[1])
    coords = pl.read_parquet(coords_path)
    is_3d = "c2" in coords.columns
    corpus = pl.read_parquet(
        "data/processed/corpus.parquet",
        columns=["award_number", "kaken_id", "title", "category"],
    )
    df = coords.join(corpus, on="award_number", how="left")
    df = df.join(load_dai_labels(), on="award_number", how="left")

    fig = go.Figure()
    for dai in [*DAI_COLORS.keys(), "複数", "区分なし"]:
        sub = df.filter(pl.col("dai") == dai)
        color = DAI_COLORS.get(dai, "#b9b8b0")
        label = f"{dai}〈{DAI_GLOSS[dai]}〉" if dai in DAI_GLOSS else dai
        # ホバーは3行: 大区分 / タイトル / 種目・課題番号（件数は凡例のみ）
        text = [
            f"{label}<br>{(t or '（タイトルなし）')[:48]}<br>{c} / {a}"
            for t, c, a in zip(sub["title"], sub["category"], sub["award_number"], strict=True)
        ]
        common = dict(
            mode="markers",
            name=f"{label} {sub.height:,}",
            text=text,
            hoverinfo="text",
            customdata=sub["kaken_id"].to_list(),  # クリックで KAKEN ページを開くのに使う
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
        legend=dict(itemsizing="constant", font=dict(size=11)),
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
    # scrollZoom: 2D でもホイール/2本指ジェスチャで拡大縮小できるようにする
    # post_script: ホバー表示中の点をクリックしたときだけ KAKEN ページを開く。
    # ダブルクリックやリセット直後の巻き込みクリック・二重発火は抑止する。
    open_kaken = (
        "var plot = document.getElementsByClassName('plotly-graph-div')[0];"
        "var hovered = null, lastK = null, lastT = 0, suppressUntil = 0;"
        "plot.on('plotly_hover', function(d){"
        "  hovered = (d.points[0] && d.points[0].customdata) || null;"
        "});"
        "plot.on('plotly_unhover', function(){ hovered = null; });"
        "plot.on('plotly_doubleclick', function(){ suppressUntil = Date.now() + 700; });"
        "plot.on('plotly_relayout', function(){ suppressUntil = Date.now() + 400; });"
        "plot.on('plotly_click', function(d){"
        "  var k = d.points[0] && d.points[0].customdata;"
        "  var now = Date.now();"
        "  if (!k || now < suppressUntil) return;"
        "  if (k !== hovered) return;"  # ホバー表示中の点だけをクリック対象にする
        "  if (k === lastK && now - lastT < 600) return;"
        "  lastK = k; lastT = now;"
        "  window.open('https://kaken.nii.ac.jp/ja/grant/' + k + '/', '_blank');"
        "});"
    )
    fig.write_html(
        out, include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"},
        post_script=open_kaken,
    )
    print(f"出力: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
