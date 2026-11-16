"""parse モジュールのテスト（実データの構造を模したフィクスチャで検証）。"""

import xml.etree.ElementTree as ET

import polars as pl

from kaken_atlas.parse import parse_award, warn_year_jumps

FIXTURE = """\
<grantAward id="KAKENHI-PROJECT-18K00001" projectType="project" awardNumber="18K00001">
  <summary xml:lang="ja">
    <title>テスト研究課題</title>
    <category path="000359" niiCode="359">基盤研究(C)</category>
    <review_section sequence="1" niiCode="100" tableType="review_section">小区分37010:分子生物学関連</review_section>
    <review_section sequence="2" niiCode="101" tableType="review_section">中区分43:分子レベルから細胞レベルの生物学およびその関連分野</review_section>
    <institution niiCode="0000001" sequence="1">テスト大学</institution>
    <member sequence="1" eradCode="10000001" role="principal_investigator"/>
    <member sequence="2" eradCode="10000002" role="co_investigator_buntan"/>
    <projectStatus fiscalYear="2020" statusCode="project_closed"/>
    <keywordList>
      <keyword>ゲノム</keyword>
      <keyword>転写制御</keyword>
    </keywordList>
    <paragraphList sequence="1" type="outline_of_research_initial">
      <paragraph sequence="1">本研究は転写制御機構を解明する。</paragraph>
      <paragraph sequence="2">新規手法を開発する。</paragraph>
    </paragraphList>
    <paragraphList sequence="2" type="outline_of_research_achievement">
      <paragraph sequence="1">機構を解明した。</paragraph>
    </paragraphList>
    <periodOfAward searchStartFiscalYear="2018" searchEndFiscalYear="2020"/>
  </summary>
  <summary xml:lang="en">
    <title>Test Project</title>
  </summary>
</grantAward>
"""


RENUMBERED = """\
<grantAward id="KAKENHI-PROJECT-23K28479" projectType="project" awardNumber="23K28479">
  <summary xml:lang="ja">
    <title>付け替えられた課題</title>
    <awardNumber awardNumber="23K28479" sequence="1"/>
    <awardNumber awardNumber="23H03791" participate="2023" sequence="2"/>
    <category path="000355" niiCode="355">基盤研究(B)</category>
    <allocation niiCode="kikin" participate="2024" sequence="1">基金</allocation>
    <allocation niiCode="hojokin" participate="2023" sequence="2">補助金</allocation>
    <projectStatus fiscalYear="2024" statusCode="granted"/>
    <periodOfAward searchStartFiscalYear="2024" searchEndFiscalYear="2025"/>
  </summary>
  <summary xml:lang="en">
    <title>Renumbered Project</title>
    <awardNumber awardNumber="23K28479" sequence="1"/>
    <awardNumber awardNumber="23H03791" participate="2023" sequence="2"/>
  </summary>
</grantAward>
"""

PUBLICLY = """\
<grantAward id="KAKENHI-PUBLICLY-21H05898" projectType="publicly" awardNumber="21H05898">
  <parentAward id="KAKENHI-AREA-20A403"><title xml:lang="ja">分子サイバネティクス</title><title xml:lang="en">Molecular Cybernetics</title></parentAward>
  <summary xml:lang="ja">
    <title>公募研究の課題</title>
    <category path="000305" niiCode="305">学術変革領域研究(A)</category>
    <periodOfAward searchStartFiscalYear="2021" searchEndFiscalYear="2022"/>
  </summary>
  <summary xml:lang="en"></summary>
</grantAward>
"""


def make_award(xml_text: str = FIXTURE) -> ET.Element:
    return ET.fromstring(xml_text)


class TestParseAward:
    def test_basic_fields(self):
        row = parse_award(make_award(), source_file="2018/test.xml")
        assert row["award_number"] == "18K00001"
        assert row["kaken_id"] == "KAKENHI-PROJECT-18K00001"
        assert row["project_type"] == "project"
        assert row["title"] == "テスト研究課題"  # 日本語 summary が優先される
        assert row["category"] == "基盤研究(C)"
        assert row["category_code"] == "359"
        assert row["source_file"] == "2018/test.xml"

    def test_review_sections_and_shokubun(self):
        row = parse_award(make_award())
        assert row["review_sections"] == [
            "小区分37010:分子生物学関連",
            "中区分43:分子レベルから細胞レベルの生物学およびその関連分野",
        ]
        # 小区分コードは「小区分NNNNN:」のものだけ抽出される
        assert row["shokubun_codes"] == ["37010"]

    def test_members(self):
        row = parse_award(make_award())
        assert row["pi_erad"] == "10000001"
        assert row["n_members"] == 2

    def test_status_and_period(self):
        row = parse_award(make_award())
        assert row["status_code"] == "project_closed"
        assert row["start_fy"] == 2018
        assert row["end_fy"] == 2020
        assert row["period_start_fy_raw"] == 2018
        assert row["old_award_numbers"] == []
        assert row["parent_award"] is None
        assert row["title_en"] == "Test Project"

    def test_renumbered_award_uses_old_number_year(self):
        """基金化で付け替えられた課題（実例 23H03791 → 23K28479 の形）。periodOfAward は基金としての期間（2024〜）になるが、
        真の開始年度は旧番号の participate（2023）。"""
        row = parse_award(make_award(RENUMBERED))
        assert row["award_number"] == "23K28479"
        assert row["start_fy"] == 2023
        assert row["period_start_fy_raw"] == 2024
        assert row["end_fy"] == 2025
        assert row["old_award_numbers"] == ["23H03791"]

    def test_participate_range_takes_first_year(self):
        xml = RENUMBERED.replace('participate="2023"', 'participate="2021-2023"')
        assert parse_award(make_award(xml))["start_fy"] == 2021

    def test_participate_later_than_period_keeps_period(self):
        # 旧番号の参加年が periodOfAward より後なら periodOfAward の開始年を使う（小さい方を取る）
        xml = RENUMBERED.replace('participate="2023"', 'participate="2026"')
        assert parse_award(make_award(xml))["start_fy"] == 2024

    def test_parent_award(self):
        row = parse_award(make_award(PUBLICLY))
        assert row["project_type"] == "publicly"
        assert row["parent_award"] == "20A403"
        assert row["parent_title"] == "分子サイバネティクス"
        assert row["title_en"] is None  # 英語 summary に題名が無い

    def test_warn_year_jumps(self):
        def block(cat, fy, n):
            return pl.DataFrame({"category": [cat] * n, "start_fy": [fy] * n})

        rows = pl.concat([
            block("基盤研究(B)", 2022, 3000), block("基盤研究(B)", 2023, 3100), block("基盤研究(B)", 2024, 11000),
            block("若手研究", 2023, 600), block("若手研究", 2024, 700),
        ])
        warnings = warn_year_jumps(rows)
        assert len(warnings) == 1 and "基盤研究(B)" in warnings[0] and "FY2024" in warnings[0]

    def test_keywords(self):
        row = parse_award(make_award())
        assert row["keywords"] == ["ゲノム", "転写制御"]

    def test_abstracts(self):
        row = parse_award(make_award())
        # 複数 paragraph は改行で結合される
        assert row["abstract_initial"] == "本研究は転写制御機構を解明する。\n新規手法を開発する。"
        assert row["abstract_achievement"] == "機構を解明した。"

    def test_en_only_summary_falls_back(self):
        xml = FIXTURE.replace('xml:lang="ja"', 'xml:lang="zz"', 1)
        row = parse_award(make_award(xml))
        assert row["title"] == "テスト研究課題"  # ja が無ければ最初の summary

    def test_no_summary_returns_none(self):
        award = ET.fromstring('<grantAward id="X" awardNumber="1"/>')
        assert parse_award(award) is None

    def test_missing_optional_fields(self):
        xml = """\
<grantAward id="KAKENHI-PROJECT-X" projectType="project" awardNumber="X">
  <summary xml:lang="ja"><title>最小限</title></summary>
</grantAward>"""
        row = parse_award(make_award(xml))
        assert row["title"] == "最小限"
        assert row["review_sections"] == []
        assert row["shokubun_codes"] == []
        assert row["pi_erad"] is None
        assert row["n_members"] == 0
        assert row["status_code"] is None
        assert row["start_fy"] is None
        assert row["keywords"] == []
        assert row["abstract_initial"] is None
