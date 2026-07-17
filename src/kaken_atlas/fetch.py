"""KAKEN opensearch API から研究課題データを取得し、生XMLを data/raw/ に保存する。

取得戦略: 年度で分割（s1=s2=年度, o1=1 = 開始年度一致）し、各年度内を
rw=500・st ページングで全件取得する。1クエリの取得上限は 200,000 件。

使い方:
    uv run python -m kaken_atlas.fetch --years 2023 --max-pages 1 --rw 20  # 試し取得
    uv run python -m kaken_atlas.fetch --years 2018-2025                   # 全件取得

注意: rw は 20/50/100/200/500 のみ有効。それ以外の値は API がエラーを返さず
黙って totalResults=0 を返す（2026-07-17 検証済み）。
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from kaken_atlas.config import DATA_RAW, load_appid

API_URL = "https://kaken.nii.ac.jp/opensearch/"
USER_AGENT = "kaken-atlas/0.1 (https://github.com/takayuki1997/kaken-atlas)"
VALID_RW = (20, 50, 100, 200, 500)
MAX_ST = 200_000
MIN_YEAR = 2018  # 小区分306は2018年度の審査区分改革で導入。それ以前は対象外。
DEFAULT_OUTDIR = DATA_RAW / "opensearch"
RETRIES = 3


def build_params(appid: str, year: int, st: int, rw: int) -> dict[str, str]:
    """年度指定・ページ指定の opensearch クエリパラメータを組み立てる。"""
    if rw not in VALID_RW:
        raise ValueError(f"rw={rw} は無効です（有効値: {VALID_RW}。仕様外の値は黙って0件になる）")
    if not 1 <= st <= MAX_ST:
        raise ValueError(f"st={st} は範囲外です（1〜{MAX_ST}）")
    if year < MIN_YEAR:
        raise ValueError(f"year={year} は対象外です（{MIN_YEAR}年度以降のみ扱う）")
    return {
        "appid": appid,
        "format": "xml",
        "s1": str(year),
        "s2": str(year),
        "o1": "1",  # 助成期間の開始年度が year に一致
        "rw": str(rw),
        "st": str(st),
    }


def parse_total_results(raw: bytes) -> int:
    """レスポンスXMLから totalResults を取り出す。エラー応答は例外にする。"""
    root = ET.fromstring(raw)
    if root.tag == "error":
        detail = root.findtext("detail") or ET.tostring(root, encoding="unicode")
        raise RuntimeError(f"KAKEN API エラー: {detail}")
    total = root.findtext("totalResults")
    if total is None:
        raise RuntimeError("レスポンスに totalResults がありません（想定外のXML構造）")
    return int(total)


def page_path(outdir: Path, year: int, st: int, rw: int) -> Path:
    return outdir / str(year) / f"st{st:07d}_rw{rw}.xml"


def fetch_page(client: httpx.Client, params: dict[str, str]) -> bytes:
    """1ページ分を取得する。一時的な失敗は指数バックオフで再試行。"""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = client.get(API_URL, params=params, timeout=60.0)
            resp.raise_for_status()
            return resp.content
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == RETRIES:
                raise
            wait = 2**attempt
            print(f"  再試行 {attempt}/{RETRIES - 1}（{wait}秒待機）: {exc}", file=sys.stderr)
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_year(
    client: httpx.Client,
    appid: str,
    year: int,
    outdir: Path,
    *,
    rw: int = 500,
    max_pages: int | None = None,
    delay: float = 1.0,
    force: bool = False,
) -> int:
    """1年度分をページングで取得し、生XMLを保存する。取得済みページはスキップ。

    Returns: その年度の totalResults。
    """
    st = 1
    pages = 0
    total: int | None = None
    while True:
        path = page_path(outdir, year, st, rw)
        if path.exists() and not force:
            raw = path.read_bytes()
            print(f"  {path.relative_to(outdir)} 取得済み・スキップ")
        else:
            raw = fetch_page(client, build_params(appid, year, st, rw))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            print(f"  {path.relative_to(outdir)} 保存（{len(raw):,} bytes）")
            time.sleep(delay)
        page_total = parse_total_results(raw)
        if page_total == 0:
            raise RuntimeError(
                f"{year}年度で totalResults=0。パラメータ不正の可能性が高い"
                "（KAKEN API は不正パラメータでも黙って0件を返す）"
            )
        if total is None:
            total = page_total
            print(f"{year}年度: 全 {total:,} 件（{-(-total // rw)} ページ）")
        pages += 1
        st += rw
        if st > total or st > MAX_ST:
            break
        if max_pages is not None and pages >= max_pages:
            print(f"  --max-pages={max_pages} に到達、{year}年度は途中で終了")
            break
    return total


def parse_years(spec: str) -> list[int]:
    """'2018-2025' / '2023' / '2018,2020,2022' を年度リストに展開する。"""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            years.extend(range(int(lo), int(hi) + 1))
        elif part:
            years.append(int(part))
    if not years:
        raise ValueError(f"年度指定を解釈できません: {spec!r}")
    return years


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KAKEN opensearch API から生XMLを取得する")
    ap.add_argument(
        "--years", default="2018-2025", help="対象年度（例: 2023, 2018-2025, 2018,2020）"
    )
    ap.add_argument("--rw", type=int, default=500, choices=VALID_RW, help="1ページの件数")
    ap.add_argument(
        "--max-pages", type=int, default=None, help="年度あたりの最大ページ数（試し取得用）"
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="保存先ディレクトリ")
    ap.add_argument("--delay", type=float, default=1.0, help="リクエスト間の待機秒数")
    ap.add_argument("--force", action="store_true", help="取得済みページも再取得する")
    args = ap.parse_args(argv)

    appid = load_appid()
    years = parse_years(args.years)
    grand_total = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for year in years:
            grand_total += fetch_year(
                client,
                appid,
                year,
                args.outdir,
                rw=args.rw,
                max_pages=args.max_pages,
                delay=args.delay,
                force=args.force,
            )
    print(f"完了: {len(years)}年度 / 合計 {grand_total:,} 件（保存先: {args.outdir}）")


if __name__ == "__main__":
    main()
