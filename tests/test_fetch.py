"""fetch モジュールの純粋関数のテスト（ネットワーク不要）。"""

import pytest

from kaken_atlas.fetch import build_params, page_path, parse_total_results, parse_years

OK_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<grantAwards>
<totalResults>16731</totalResults>
<startIndex>1</startIndex>
<itemsPerPage>20</itemsPerPage>
</grantAwards>"""

ERROR_XML = b"""<?xml version="1.0"?>
<error>
    <code>403</code>
    <reason>Forbidden</reason>
    <detail>Invalid APPID</detail>
</error>"""


class TestBuildParams:
    def test_valid(self):
        params = build_params("myid", 2023, 501, 500)
        assert params["s1"] == params["s2"] == "2023"
        assert params["o1"] == "1"
        assert params["st"] == "501"
        assert params["format"] == "xml"

    def test_invalid_rw_rejected(self):
        # rw=1 などの仕様外の値は API が黙って0件を返すため、手前で弾く
        with pytest.raises(ValueError, match="rw=1"):
            build_params("myid", 2023, 1, 1)

    def test_st_out_of_range(self):
        with pytest.raises(ValueError, match="st="):
            build_params("myid", 2023, 200_001, 500)
        with pytest.raises(ValueError, match="st="):
            build_params("myid", 2023, 0, 500)

    def test_pre_2018_rejected(self):
        with pytest.raises(ValueError, match="2018"):
            build_params("myid", 2017, 1, 500)


class TestParseTotalResults:
    def test_ok(self):
        assert parse_total_results(OK_XML) == 16731

    def test_api_error_raises(self):
        with pytest.raises(RuntimeError, match="Invalid APPID"):
            parse_total_results(ERROR_XML)

    def test_missing_total_raises(self):
        with pytest.raises(RuntimeError, match="totalResults"):
            parse_total_results(b"<grantAwards></grantAwards>")


class TestParseYears:
    def test_single(self):
        assert parse_years("2023") == [2023]

    def test_range(self):
        assert parse_years("2018-2021") == [2018, 2019, 2020, 2021]

    def test_list(self):
        assert parse_years("2018,2020,2022") == [2018, 2020, 2022]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_years("")


def test_page_path(tmp_path):
    path = page_path(tmp_path, 2023, 501, 500)
    assert path == tmp_path / "2023" / "st0000501_rw500.xml"
