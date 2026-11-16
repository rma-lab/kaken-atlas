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
YEAR_RE = re.compile(r"\d{4}")
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
    "period_start_fy_raw": pl.Int32,
    "old_award_numbers": pl.List(pl.Utf8),
    "parent_award": pl.Utf8,
    "parent_title": pl.Utf8,
    "title_en": pl.Utf8,
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


def _title_with_fallback(award: ET.Element, summary: ET.Element) -> str | None:
    """タイトルは日本語 summary を優先し、無ければ他言語（英語）の summary から補完する。

    英語で応募された課題は ja summary にタイトルが無いことがある（2,279件確認済み）。
    """
    title = summary.findtext("title")
    if title:
        return title
    for s in award.findall("summary"):
        if title := s.findtext("title"):
            return title
    return None


def _start_fy(summary: ET.Element, period_start: int | None) -> tuple[int | None, list[str]]:
    """開始年度と旧課題番号を返す。

    補助金から基金に移った継続課題（令和 5 年度補正予算で基盤研究(B) など。
    例 23H03791 → 23K28479）は KAKEN 上で新番号の別レコードになり、summary 直下の
    periodOfAward は「基金としての期間」（切り替え年度から）になる。真の開始年度は、
    periodOfAward の開始年と、旧課題番号（awardNumber の sequence≠1）の participate
    属性に現れる最初の年のうち小さい方（ルール R1。全件検証は
    reports/handoff/2026-11-13_start_fy_bug.md）。participate は「2023」「2024-2025」
    「2021, 2023」の形なので最初の 4 桁を年とみなす。
    """
    old_numbers: list[str] = []
    years: list[int] = []
    for an in summary.findall("awardNumber"):
        if an.get("sequence") == "1" or not an.get("awardNumber"):
            continue
        old_numbers.append(an.get("awardNumber"))
        if m := YEAR_RE.search(an.get("participate") or ""):
            years.append(int(m.group()))
    if period_start is not None:
        years.append(period_start)
    return (min(years) if years else None), old_numbers


def _title_en(award: ET.Element) -> str | None:
    """英語 summary の題名（KAKEN の公式英題。約 2/3 の課題にある）。"""
    for s in award.findall("summary"):
        if s.get(XML_LANG) == "en":
            return s.findtext("title") or None
    return None


def _parent(award: ET.Element) -> tuple[str | None, str | None]:
    """学術変革領域研究などの親課題（領域）。

    grantAward 直下の parentAward（id は KAKENHI-AREA-20A403 の形）から領域番号と和文題名を取る。
    """
    p = award.find("parentAward")
    if p is None:
        return None, None
    pid = p.get("id") or ""
    number = pid.rsplit("-", 1)[-1] if pid else None
    title = next((t.text for t in p.findall("title") if t.get(XML_LANG) == "ja" and t.text), None)
    if title is None:
        title = p.findtext("title")
    return number or None, title


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
    period_start = _int_or_none(period.get("searchStartFiscalYear")) if period is not None else None
    end_fy = period.get("searchEndFiscalYear") if period is not None else None
    start_fy, old_numbers = _start_fy(summary, period_start)
    parent_award, parent_title = _parent(award)
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
        "title": _title_with_fallback(award, summary),
        "category": category.text if category is not None else None,
        "category_code": category.get("niiCode") if category is not None else None,
        "review_sections": review_sections,
        "shokubun_codes": shokubun_codes,
        "institution": summary.findtext("institution"),
        "pi_erad": pi_erad,
        "n_members": len(members),
        "status_code": status.get("statusCode") if status is not None else None,
        "start_fy": start_fy,
        "end_fy": _int_or_none(end_fy),
        "period_start_fy_raw": period_start,
        "old_award_numbers": old_numbers,
        "parent_award": parent_award,
        "parent_title": parent_title,
        "title_en": _title_en(award),
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
    moved = df.filter(pl.col("start_fy") != pl.col("period_start_fy_raw")).height
    n_parent = df["parent_award"].is_not_null().sum()
    n_en = df["title_en"].is_not_null().sum()
    print(f"  開始年度を旧課題番号で補正: {moved:,} 件")
    print(f"  親課題あり: {n_parent:,} / 英題あり: {n_en:,}")
    warn_year_jumps(df)


def warn_year_jumps(df: pl.DataFrame, factor: float = 2.0, min_count: int = 500) -> list[str]:
    """種目×開始年度の件数が前年の factor 倍を超えたら警告する。

    基金化の付け替えで基盤研究(B) が FY2024 に 3 倍集まった偏りの再発防止。
    """
    counts = (
        df.filter(pl.col("start_fy").is_not_null())
        .group_by("category", "start_fy").len().sort("category", "start_fy")
    )
    warnings: list[str] = []
    prev: dict[str, tuple[int, int]] = {}
    for cat, fy, n in counts.iter_rows():
        if cat in prev and prev[cat][0] == fy - 1 and n >= min_count and n > factor * prev[cat][1]:
            prev_n = prev[cat][1]
            warnings.append(
                f"  警告: {cat} の FY{fy} は {n:,} 件で前年の {prev_n:,} 件の {factor:g} 倍超"
            )
        prev[cat] = (fy, n)
    for w in warnings:
        print(w, file=sys.stderr)
    return warnings


if __name__ == "__main__":
    main()
