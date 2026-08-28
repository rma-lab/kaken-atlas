"""審査区分表（data/reference/kubun_table.csv）を使う共通ヘルパ。

- 小区分コード → 大区分集合の辞書
- 課題（小区分コードのリスト）→ 大区分ラベルの分類
  （一意→その大区分 / 複数にまたがる→「複数」/ 小区分なし→「区分なし」）
- 大区分の便宜的通称と、意味順色相環の11色（導出は scripts/plot_map_dai.py 参照）
"""

from __future__ import annotations

import re
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


AWARDS_PARQUET = Path("data/interim/awards.parquet")

_RE_SHO = re.compile(r"小区分(\d{5})")
_RE_CHU = re.compile(r"中区分(\d+):")
_RE_DAI = re.compile(r"大区分([A-K])$")


def load_kubun_maps(path: Path = KUBUN_CSV) -> tuple[dict[str, set[str]], dict[int, set[str]]]:
    """小区分コード→大区分集合、中区分番号→大区分集合 の2辞書を返す。"""
    tab = pl.read_csv(path, schema_overrides={"sho_code": pl.Utf8})
    sho2dai: dict[str, set[str]] = {}
    chu2dai: dict[int, set[str]] = {}
    for r in tab.iter_rows(named=True):
        sho2dai.setdefault(r["sho_code"], set()).add(r["dai_code"])
        chu2dai.setdefault(r["chu_code"], set()).add(r["dai_code"])
    return sho2dai, chu2dai


def classify_review_sections(
    review_sections, sho2dai: dict[str, set[str]], chu2dai: dict[int, set[str]]
) -> str:
    """審査区分の文字列リストから大区分ラベルを決める。

    小区分（基盤B/C・若手等）・中区分（基盤A・挑戦的等）・大区分直接（基盤S）を解釈。
    一意→その大区分 / 複数にまたがる→「複数」/ 判別不能（スタート支援の独自区分・
    新学術等の区分なし）→「区分なし」。
    """
    dais: set[str] = set()
    for rs in review_sections:
        if m := _RE_SHO.match(rs):
            dais |= sho2dai.get(m.group(1), set())
        elif m := _RE_CHU.match(rs):
            dais |= chu2dai.get(int(m.group(1)), set())
        elif m := _RE_DAI.match(rs):
            dais.add(m.group(1))
    if not dais:
        return "区分なし"
    return next(iter(dais)) if len(dais) == 1 else "複数"


def load_dai_labels() -> pl.DataFrame:
    """全課題の award_number → 大区分ラベル（dai 列）の対応表を返す。"""
    awards = pl.read_parquet(AWARDS_PARQUET, columns=["award_number", "review_sections"])
    sho2dai, chu2dai = load_kubun_maps()
    labels = [
        classify_review_sections(rs.to_list(), sho2dai, chu2dai)
        for rs in awards["review_sections"]
    ]
    return awards.select("award_number").with_columns(dai=pl.Series(labels))
