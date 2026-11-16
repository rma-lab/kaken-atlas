"""awards.parquet から埋め込み用コーパス（corpus.parquet）を構築する。

フィルタ条件（確定事項、CLAUDE.md 参照）:
- 開始年度 2019〜2025（2018年度は採択時概要が無いため除外。2026-08決定）。
  **絞り込みは parse 時の生の開始年度（period_start_fy_raw）で行う。** 基金化の付け替えで
  start_fy を補正しても（2026-11、8,923 件）行の集合と順序を変えないため（embeddings.parquet と
  行対応している）。補正後に 2018 年度以前となる課題も採択時概要があるので残す。
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
    "period_start_fy_raw",
    "old_award_numbers",
    "parent_award",
    "title_en",
]


def build_corpus(awards: pl.DataFrame) -> pl.DataFrame:
    """確定フィルタを適用し、埋め込みテキスト列 text を組み立てる。"""
    fy_col = "period_start_fy_raw" if "period_start_fy_raw" in awards.columns else "start_fy"
    carry = [c for c in CARRY_COLUMNS if c in awards.columns]
    filtered = awards.filter(
        pl.col(fy_col).is_between(START_FY_MIN, START_FY_MAX),
        # ne_missing: null（ステータス欠損）を落とさず declined のみ除外する
        pl.col("status_code").ne_missing("declined"),
        pl.col("abstract_initial").is_not_null(),
    )
    return filtered.select(
        *carry,
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
    ap.add_argument(
        "--keep-embedded-text", action="store_true",
        help="text と n_tokens を既存の出力から引き継ぐ（埋め込みはその text から計算されている）。"
        "新しい text と異なる行は data/interim/corpus_text_drift.parquet に記録する",
    )
    ap.add_argument(
        "--check-order", action="store_true",
        help="既存の出力と award_number の並びが完全一致することを検査（埋め込みとの行対応の保護）",
    )
    args = ap.parse_args(argv)

    awards = pl.read_parquet(args.in_path)
    corpus = build_corpus(awards)
    print(f"フィルタ後: {corpus.height:,} 件（元 {awards.height:,} 件）")

    if args.check_order or args.keep_embedded_text:
        old = pl.read_parquet(args.out)
        same = old.height == corpus.height and (
            old["award_number"] == corpus["award_number"]
        ).all()
        if not same:
            raise SystemExit("award_number の並びが既存の corpus.parquet と一致しません")
        print("  並びの検査: 既存の corpus.parquet と完全一致")
        if args.keep_embedded_text:
            drift = corpus.filter(corpus["text"] != old["text"]).select("award_number", "title")
            if drift.height:
                drift_path = DATA_INTERIM / "corpus_text_drift.parquet"
                drift.write_parquet(drift_path)
                print(
                    f"  text が新旧で異なる行: {drift.height:,} 件（埋め込み時の text を保持。"
                    f"次回の埋め込み更新で解消。一覧 {drift_path}）"
                )
            corpus = corpus.with_columns(text=old["text"], n_tokens=old["n_tokens"])
            print("  text・n_tokens: 既存（埋め込み時）の値を引き継ぎ")
    elif not args.no_tokens:
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
