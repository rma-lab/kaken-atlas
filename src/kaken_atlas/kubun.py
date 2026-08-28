"""審査区分表（data/reference/kubun_table.csv）を使う共通ヘルパ。

- 小区分コード → 大区分集合の辞書
- 課題（小区分コードのリスト）→ 大区分ラベルの分類
  （一意→その大区分 / 複数にまたがる→「複数」/ 小区分なし→「区分なし」）
- 大区分の便宜的通称と、意味順色相環の11色（導出は scripts/plot_map_dai.py 参照）
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

KUBUN_CSV = Path("data/reference/kubun_table.csv")

# 意味順色相環（OKLCH L=0.60, C=0.17。巡回路 A→J→C→B→D→E→K→F→G→H→I→A に等間隔）
DAI_COLORS = {
    "A": "#bf50a0", "B": "#0088db", "C": "#6473e4", "D": "#0098b6",
    "E": "#009f7d", "F": "#868700", "G": "#b66f00", "H": "#ce5601",
    "I": "#d14a63", "J": "#9b5fce", "K": "#2e9932",
}
DAI_GLOSS = {
    "A": "人文学・社会科学", "B": "数物系科学", "C": "化学", "D": "工学",
    "E": "材料・応用工学", "F": "農学", "G": "生物学", "H": "薬学・基礎医学",
    "I": "医歯薬学（臨床）", "J": "情報学", "K": "環境学",
}


def load_sho2dai(path: Path = KUBUN_CSV) -> dict[str, set[str]]:
    tab = pl.read_csv(path, schema_overrides={"sho_code": pl.Utf8})
    sho2dai: dict[str, set[str]] = {}
    for r in tab.iter_rows(named=True):
        sho2dai.setdefault(r["sho_code"], set()).add(r["dai_code"])
    return sho2dai


def classify_dai(shokubun_codes, sho2dai: dict[str, set[str]]) -> str:
    dais: set[str] = set()
    for c in shokubun_codes:
        dais |= sho2dai.get(c, set())
    if not dais:
        return "区分なし"
    return next(iter(dais)) if len(dais) == 1 else "複数"
