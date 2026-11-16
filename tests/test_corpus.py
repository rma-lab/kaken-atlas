"""corpus.build_corpus のフィルタ・テキスト組み立てのテスト。"""

import polars as pl

from kaken_atlas.corpus import build_corpus

SCHEMA_OVERRIDES = {"keywords": pl.List(pl.Utf8), "shokubun_codes": pl.List(pl.Utf8)}


def _award(**overrides) -> dict:
    base = {
        "award_number": "19K00001",
        "kaken_id": "KAKENHI-PROJECT-19K00001",
        "title": "テスト課題",
        "category": "基盤研究(C)",
        "shokubun_codes": ["37010"],
        "status_code": "granted",
        "start_fy": 2019,
        "end_fy": 2021,
        "keywords": ["キーワードA", "キーワードB"],
        "abstract_initial": "研究概要の本文。",
        "abstract_achievement": None,
    }
    return base | overrides


def test_filters_and_text():
    awards = pl.DataFrame(
        [
            _award(),
            _award(award_number="18K00001", start_fy=2018),  # 2018年度開始 → 除外
            _award(award_number="19K00002", status_code="declined"),  # 不採択 → 除外
            _award(award_number="19K00003", abstract_initial=None),  # 概要なし → 除外
            _award(award_number="19K00004", status_code=None),  # ステータス欠損 → 残す
        ],
        schema_overrides=SCHEMA_OVERRIDES,
    )
    corpus = build_corpus(awards)
    assert corpus["award_number"].to_list() == ["19K00001", "19K00004"]
    assert corpus["text"][0] == "テスト課題\nキーワードA、キーワードB\n研究概要の本文。"


def test_filter_uses_raw_period_year_and_keeps_corrected_start_fy():
    """基金化で付け替えられた課題: 生の開始年度（2024）で絞り込み、start_fy には補正後（2023）を持たせる。
    補正後に 2018 年度以前になる課題も、生の年度が範囲内なら残す（行の集合を変えない）。"""
    awards = pl.DataFrame(
        [
            _award(award_number="23K28479", start_fy=2023, period_start_fy_raw=2024,
                   old_award_numbers=["23H03791"]),
            _award(award_number="22K99999", start_fy=2018, period_start_fy_raw=2022, old_award_numbers=["18H9"]),
            _award(award_number="18K00001", start_fy=2018, period_start_fy_raw=2018, old_award_numbers=[]),
        ],
        schema_overrides=SCHEMA_OVERRIDES | {"old_award_numbers": pl.List(pl.Utf8)},
    )
    corpus = build_corpus(awards)
    assert corpus["award_number"].to_list() == ["23K28479", "22K99999"]
    assert corpus["start_fy"].to_list() == [2023, 2018]
    assert corpus["period_start_fy_raw"].to_list() == [2024, 2022]
    assert corpus["old_award_numbers"][0].to_list() == ["23H03791"]


def test_empty_keywords_do_not_break_text():
    corpus = build_corpus(
        pl.DataFrame([_award(keywords=[])], schema_overrides=SCHEMA_OVERRIDES)
    )
    assert corpus["text"][0] == "テスト課題\n\n研究概要の本文。"
