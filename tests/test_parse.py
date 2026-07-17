"""parse モジュールのテスト（実データの構造を模したフィクスチャで検証）。"""

import xml.etree.ElementTree as ET

from kaken_atlas.parse import parse_award

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
