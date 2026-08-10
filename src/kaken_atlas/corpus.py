"""awards.parquet から埋め込み用コーパス（corpus.parquet）を構築する。

フィルタ条件（確定事項、CLAUDE.md 参照）:
- 開始年度 2019〜2025（2018年度は採択時概要が無いため除外。2026-08決定）
- declined（不採択）を除外。それ以外のステータス（granted/project_closed/
  discontinued 等）は実際に採択された課題なので残す。
- 採択時概要（abstract_initial）があるもののみ。

埋め込みテキストはタイトル・キーワード・採択時概要の改行連結。
Ruri v3 のクラスタリング用途では空プレフィックス（意味エンコード）モードを
使うため、プレフィックスは付けない。トークン数は Ruri v3 のトークナイザで
実測し n_tokens 列に持たせる（最大8192の上限チェックは下流で行える）。

使い方:
    uv run python -m kaken_atlas.corpus                # → data/processed/corpus.parquet
    uv run python -m kaken_atlas.corpus --no-tokens    # トークン数算出をスキップ（高速）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from kaken_atlas.config import DATA_INTERIM, DATA_PROCESSED

DEFAULT_IN = DATA_INTERIM / "awards.parquet"
DEFAULT_OUT = DATA_PROCESSED / "corpus.parquet"

TOKENIZER_MODEL = "cl-nagoya/ruri-v3-310m"
START_FY_MIN = 2019
START_FY_MAX = 2025

# 下流（クラスタリング・指標計算）で使う属性列。テキスト系以外は awards.parquet
# から award_number で引けるが、頻用するものは持ち回る。
CARRY_COLUMNS = [
    "award_number",
    "kaken_id",
    "title",
    "category",
    "shokubun_codes",
    "status_code",
    "start_fy",
    "end_fy",
]


def build_corpus(awards: pl.DataFrame) -> pl.DataFrame:
    """確定フィルタを適用し、埋め込みテキスト列 text を組み立てる。"""
    filtered = awards.filter(
        pl.col("start_fy").is_between(START_FY_MIN, START_FY_MAX),
        # ne_missing: null（ステータス欠損）を落とさず declined のみ除外する
        pl.col("status_code").ne_missing("declined"),
        pl.col("abstract_initial").is_not_null(),
    )
    return filtered.select(
        *CARRY_COLUMNS,
        text=pl.concat_str(
            pl.col("title"),
            pl.col("keywords").list.join("、"),
            pl.col("abstract_initial"),
            separator="\n",
            ignore_nulls=True,
        ),
    )


def add_token_counts(corpus: pl.DataFrame, batch_size: int = 2048) -> pl.DataFrame:
    """Ruri v3 トークナイザで text のトークン数（特殊トークン込み）を実測する。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    texts = corpus["text"].to_list()
    counts: list[int] = []
    for i in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[i : i + batch_size], truncation=False)
        counts.extend(len(ids) for ids in encoded["input_ids"])
        done = min(i + batch_size, len(texts))
        if done % (batch_size * 10) == 0 or done == len(texts):
            print(f"  トークン数算出: {done:,}/{len(texts):,}")
    return corpus.with_columns(n_tokens=pl.Series(counts, dtype=pl.Int32))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="埋め込み用コーパスを構築する")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-tokens", action="store_true", help="トークン数の算出をスキップ")
    args = ap.parse_args(argv)

    awards = pl.read_parquet(args.in_path)
    corpus = build_corpus(awards)
    print(f"フィルタ後: {corpus.height:,} 件（元 {awards.height:,} 件）")

    if not args.no_tokens:
        corpus = add_token_counts(corpus)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_parquet(args.out)
    print(f"完了: {corpus.height:,} 件 → {args.out}")

    if "n_tokens" in corpus.columns:
        stats = corpus["n_tokens"]
        print(
            f"  n_tokens: min={stats.min()} / median={stats.median():.0f} "
            f"/ p99={stats.quantile(0.99):.0f} / max={stats.max()}"
        )
        over = corpus.filter(pl.col("n_tokens") > 8192).height
        print(f"  8192トークン超: {over:,} 件")


if __name__ == "__main__":
    main()
