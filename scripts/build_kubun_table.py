"""公式マスタ XML から科研費審査区分表（大・中・小）を機械可読形式で生成する。

出所:
- 階層・コード・名前（日英）: niijp/grants_masterxml_kaken の review_section_master_kakenhi.xml
  （type="review_section", start_date="2018-04-01" のテーブル。M付き合同審査区分は除外）
- 小区分の内容キーワード: 検証済みの KubunTable.csv（2026-08-28 に公式マスタと全数照合済み）

出力（--outdir 以下）:
- kubun_table.csv  : tidy 形式（1行 = 小区分の1所属、323行、UTF-8）
- kubun_table.json : 小区分コードをキーにした辞書（所属リスト・キーワード付き、UTF-8）

使い方:
    uv run python scripts/build_kubun_table.py --outdir data/interim
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import polars as pl

MASTER_URL = (
    "https://bitbucket.org/niijp/grants_masterxml_kaken/raw/HEAD/review_section_master_kakenhi.xml"
)
MASTER_CACHE = Path("data/raw/masters/review_section_master_kakenhi.xml")
KEYWORDS_CSV = Path(
    "/Users/takayuki/Library/CloudStorage/Dropbox/研究IR/区分分類/BERT/KubunTable.csv"
)


def _name(el: ET.Element, lang: str) -> str:
    text = el.findtext(f"name[@lang='{lang}']") or ""
    # "中区分1:思想、…" / "小区分01010:哲学…" の接頭辞を落として素の名前にする
    prefix = (
        r"^(大区分|中区分\d+:|小区分\d+:"
        r"|Broad Section |Medium-sized Section \d+:|Basic Section \d+:)"
    )
    return re.sub(prefix, "", text).strip()


def load_master() -> list[dict]:
    if not MASTER_CACHE.exists():
        MASTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MASTER_CACHE.write_bytes(httpx.get(MASTER_URL, follow_redirects=True, timeout=30).content)
    root = ET.parse(MASTER_CACHE).getroot()
    rows = []
    for table in root.findall("review_section_table"):
        if table.get("type") != "review_section":
            continue
        for dai in table.findall("review_section"):
            dai_code = dai.findtext("code[@type='mext']")
            for chu in dai.findall("review_section"):
                chu_code = int(chu.findtext("code[@type='mext']"))
                for sho in chu.findall("review_section"):
                    sho_code = sho.findtext("code[@type='mext']")
                    if not re.fullmatch(r"\d{5}", sho_code):
                        continue  # M付き合同審査区分は実データに現れないため除外
                    rows.append(
                        {
                            "dai_code": dai_code,
                            "chu_code": chu_code,
                            "chu_name_ja": _name(chu, "ja"),
                            "chu_name_en": _name(chu, "en"),
                            "sho_code": sho_code,
                            "sho_name_ja": _name(sho, "ja"),
                            "sho_name_en": _name(sho, "en"),
                        }
                    )
    return rows


def load_keywords() -> dict[str, str]:
    tab = pl.read_csv(KEYWORDS_CSV, encoding="cp932")
    tab = tab.with_columns(sho=pl.col("tabSho").cast(pl.Utf8).str.zfill(5))
    return dict(zip(tab["sho"], tab["tabShoCon"], strict=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="審査区分表を機械可読形式で生成")
    ap.add_argument("--outdir", type=Path, default=Path("data/reference"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = load_master()
    keywords = load_keywords()
    df = pl.DataFrame(rows).sort(["dai_code", "chu_code", "sho_code"])
    df = df.with_columns(
        sho_keywords=pl.col("sho_code").replace_strict(keywords, default=""),
        n_memberships=pl.len().over("sho_code"),
    )

    csv_path = args.outdir / "kubun_table.csv"
    df.write_csv(csv_path)
    print(f"CSV : {csv_path} ({df.height} 行, 小区分 {df['sho_code'].n_unique()} 種)")

    by_sho: dict[str, dict] = {}
    for r in df.iter_rows(named=True):
        entry = by_sho.setdefault(
            r["sho_code"],
            {
                "name_ja": r["sho_name_ja"],
                "name_en": r["sho_name_en"],
                "keywords": r["sho_keywords"],
                "memberships": [],
            },
        )
        entry["memberships"].append(
            {"dai_code": r["dai_code"], "chu_code": r["chu_code"], "chu_name_ja": r["chu_name_ja"]}
        )
    json_path = args.outdir / "kubun_table.json"
    json_path.write_text(json.dumps(by_sho, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"JSON: {json_path} ({len(by_sho)} 小区分)")


if __name__ == "__main__":
    main()
