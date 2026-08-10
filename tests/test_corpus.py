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


def test_empty_keywords_do_not_break_text():
    corpus = build_corpus(
        pl.DataFrame([_award(keywords=[])], schema_overrides=SCHEMA_OVERRIDES)
    )
    assert corpus["text"][0] == "テスト課題\n\n研究概要の本文。"
