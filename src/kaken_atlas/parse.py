"""data/raw/opensearch の生XMLをパースし、構造化テーブル（parquet）を出力する。

各 grantAward から課題番号・タイトル・審査区分・研究概要などを抽出する。
フィルタリング（declined 除外・種目絞り込み等）はここでは行わず、
status_code などの列を持たせて下流の分析側で行う方針。

使い方:
    uv run python -m kaken_atlas.parse                        # 全ファイル → parquet
    uv run python -m kaken_atlas.parse --limit-files 2        # 動作確認用
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

from kaken_atlas.config import DATA_INTERIM, DATA_RAW

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
SHOKUBUN_RE = re.compile(r"^小区分(\d{5}):")
DEFAULT_RAW_DIR = DATA_RAW / "opensearch"
DEFAULT_OUT = DATA_INTERIM / "awards.parquet"

SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "award_number": pl.Utf8,
    "kaken_id": pl.Utf8,
    "project_type": pl.Utf8,
    "title": pl.Utf8,
    "category": pl.Utf8,
    "category_code": pl.Utf8,
    "review_sections": pl.List(pl.Utf8),
    "shokubun_codes": pl.List(pl.Utf8),
    "institution": pl.Utf8,
    "pi_erad": pl.Utf8,
    "n_members": pl.Int32,
    "status_code": pl.Utf8,
    "start_fy": pl.Int32,
    "end_fy": pl.Int32,
    "keywords": pl.List(pl.Utf8),
    "abstract_initial": pl.Utf8,
    "abstract_achievement": pl.Utf8,
    "source_file": pl.Utf8,
}


def _ja_summary(award: ET.Element) -> ET.Element | None:
    """summary は言語ごとに複数ある。日本語を優先し、なければ最初のものを返す。"""
    summaries = award.findall("summary")
    for s in summaries:
        if s.get(XML_LANG) == "ja":
            return s
    return summaries[0] if summaries else None


def _paragraph_text(summary: ET.Element, ptype: str) -> str | None:
    """指定 type の paragraphList 内の段落を結合して返す。"""
    for pl_el in summary.findall("paragraphList"):
        if pl_el.get("type") == ptype:
            text = "\n".join((p.text or "").strip() for p in pl_el.findall("paragraph"))
            return text.strip() or None
    return None


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value else None


def parse_award(award: ET.Element, source_file: str = "") -> dict | None:
    """1課題（grantAward 要素）をフラットな dict にする。summary が無ければ None。"""
    summary = _ja_summary(award)
    if summary is None:
        return None

    review_sections = [
        (r.text or "").strip() for r in summary.findall("review_section") if (r.text or "").strip()
    ]
    shokubun_codes = [
        m.group(1) for rs in review_sections if (m := SHOKUBUN_RE.match(rs))
    ]

    category = summary.find("category")
    members = summary.findall("member")
    pi_erad = next(
        (m.get("eradCode") for m in members if m.get("role") == "principal_investigator"),
        None,
    )
    status = summary.find("projectStatus")
    period = summary.find("periodOfAward")
    start_fy = period.get("searchStartFiscalYear") if period is not None else None
    end_fy = period.get("searchEndFiscalYear") if period is not None else None
    keyword_list = summary.find("keywordList")
    keywords = (
        [(k.text or "").strip() for k in keyword_list if (k.text or "").strip()]
        if keyword_list is not None
        else []
    )

    return {
        "award_number": award.get("awardNumber"),
        "kaken_id": award.get("id"),
        "project_type": award.get("projectType"),
        "title": summary.findtext("title"),
        "category": category.text if category is not None else None,
        "category_code": category.get("niiCode") if category is not None else None,
        "review_sections": review_sections,
        "shokubun_codes": shokubun_codes,
        "institution": summary.findtext("institution"),
        "pi_erad": pi_erad,
        "n_members": len(members),
        "status_code": status.get("statusCode") if status is not None else None,
        "start_fy": _int_or_none(start_fy),
        "end_fy": _int_or_none(end_fy),
        "keywords": keywords,
        "abstract_initial": _paragraph_text(summary, "outline_of_research_initial"),
        "abstract_achievement": _paragraph_text(summary, "outline_of_research_achievement"),
        "source_file": source_file,
    }


def parse_file(path: Path, raw_dir: Path | None = None) -> list[dict]:
    """1ページ分のXMLファイルをパースする。"""
    rel = str(path.relative_to(raw_dir)) if raw_dir else path.name
    root = ET.parse(path).getroot()
    rows = []
    for award in root.iter("grantAward"):
        row = parse_award(award, source_file=rel)
        if row is not None:
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KAKEN 生XMLを parquet に構造化する")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit-files", type=int, default=None, help="先頭Nファイルのみ（動作確認用）")
    args = ap.parse_args(argv)

    files = sorted(args.raw_dir.rglob("*.xml"))
    if not files:
        sys.exit(f"XMLファイルが見つかりません: {args.raw_dir}")
    if args.limit_files:
        files = files[: args.limit_files]

    rows: list[dict] = []
    for i, path in enumerate(files, 1):
        rows.extend(parse_file(path, raw_dir=args.raw_dir))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ファイル / {len(rows):,} 件")

    df = pl.DataFrame(rows, schema=SCHEMA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)
    print(f"完了: {df.height:,} 件 → {args.out}")

    # 内訳の要約（目視での妥当性チェック用）
    with_abst = df.filter(pl.col("abstract_initial").is_not_null()).height
    with_shokubun = df.filter(pl.col("shokubun_codes").list.len() > 0).height
    print(f"  研究概要あり: {with_abst:,} / 小区分あり: {with_shokubun:,}")
    print(df.group_by("status_code").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
