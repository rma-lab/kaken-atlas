"""公式マスタ XML から科研費審査区分表（大・中・小）を機械可読形式で生成する。

出所:
- 階層・コード・名前（日英）: niijp/grants_masterxml_kaken の review_section_master_kakenhi.xml
  （type="review_section", start_date="2018-04-01" のテーブル。M付き合同審査区分は除外）
- 小区分の「内容の例」（現行）: data/reference/shinsa_kubun_r4.json（JSPS 審査区分表 別表2、令和4年3月9日決定・令和5年度審査から適用。
  令和9年度公募ページ配布の PDF から抽出し、NII マスタと突合したもの。whet プロジェクトで 2026-09-23 作成）
- 2018 年初版の「内容の例」: 手作りの KubunTable.csv（参考として sho_keywords_2018 列に残す。令和4年改正で 99 小区分の語彙が見直された）

出力（--outdir 以下）:
- kubun_table.csv  : tidy 形式（1行 = 小区分の1所属、323行、UTF-8）。sho_keywords＝現行、sho_keywords_2018＝初版
- kubun_table.json : 小区分コードをキーにした辞書（所属リスト・keywords〔現行〕・keywords_2018 付き、UTF-8）

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
EXAMPLES_JSON = Path("data/reference/shinsa_kubun_r4.json")  # 現行の「内容の例」


def load_examples() -> dict[str, str]:
    """現行（令和4年改正）の小区分の「内容の例」。小区分コード → 文字列。"""
    d = json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))
    return {it["code"]: it["examples"] for it in d["shokubun"]}


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
    keywords_2018 = load_keywords()
    examples = load_examples()
    df = pl.DataFrame(rows).sort(["dai_code", "chu_code", "sho_code"])
    df = df.with_columns(
        sho_keywords=pl.col("sho_code").replace_strict(examples, default=""),
        sho_keywords_2018=pl.col("sho_code").replace_strict(keywords_2018, default=""),
        n_memberships=pl.len().over("sho_code"),
    )
    n_diff = (df.filter(pl.col("sho_keywords").str.replace_all(" ", "") != pl.col("sho_keywords_2018").str.replace_all(" ", ""))["sho_code"].n_unique())
    print(f"内容の例が 2018 年版と異なる小区分: {n_diff}")

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
                "keywords_2018": r["sho_keywords_2018"],
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
