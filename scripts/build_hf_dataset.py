"""Hugging Face Datasets 用の公開データ一式を組み立てる（rma-lab/kaken-atlas-embeddings）。

入れるもの: 課題番号、Ruri v3 の埋め込み（768 次元）、タイトル・キーワード・種目・年度（地図サイトで既に公開している範囲）、
2D/3D/球面の座標、内容由来の色、キーワード地名の山域、768 次元の最近傍距離、色の輪のノード、地名と凡例の JSON。
入れないもの: 研究概要の本文（課題番号から KAKEN で引ける。一括再配布はしない）、研究者名・所属機関（サイトの方針）、配分額。

行順は全ファイルで corpus.parquet と同一（課題番号で結合もできる）。

出力: data/processed/hf/kaken-atlas-embeddings/
    embeddings/train-0000i-of-0000N.parquet   … award_number, embedding（float32×768、L2 正規化済み）
    metadata/train-00000-of-00001.parquet     … 属性・座標・色・地名
    extras/placenames_2d.json, color_legend.json, elastic_ring_nodes.npz
    README.md は scripts/hf/dataset_card.md から複写（件数などを差し込む）

使い方:
    uv run python scripts/build_hf_dataset.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from kaken_atlas.kubun import DAI_GLOSS, load_dai_labels  # noqa: E402

P = Path("data/processed")
OUT = P / "hf" / "kaken-atlas-embeddings"
CARD = Path("scripts/hf/dataset_card.md")
N_SHARDS = 4


def main() -> None:
    corpus = pl.read_parquet(P / "corpus.parquet", columns=[
        "award_number", "kaken_id", "category", "shokubun_codes", "start_fy", "end_fy", "n_tokens"])
    n = corpus.height
    keys = corpus["award_number"]

    def aligned(path: Path, cols: list[str]) -> pl.DataFrame:
        df = pl.read_parquet(path, columns=["award_number", *cols])
        assert df.height == n and bool((df["award_number"] == keys).all()), f"{path}: 行順が corpus と違う"
        return df.drop("award_number")

    # タイトルは awards.parquet から（英語のみの課題のタイトルを補完済み。corpus 側は 2,098 件が null）
    kw = pl.read_parquet("data/interim/awards.parquet", columns=["award_number", "title", "keywords"])
    dai = load_dai_labels()
    places = json.loads((P / "placenames_2d.json").read_text(encoding="utf-8"))
    name_of = [{p["id"]: "・".join(p["words"]) for p in lv["places"]} for lv in places["levels"]]
    sig = [f"{lv['sigma']:g}" for lv in places["levels"]]
    assert sig == ["0.3", "0.15"], sig

    u2 = aligned(P / "umap2d_nn15_md0.1.parquet", ["c0", "c1"]).rename({"c0": "umap2d_x", "c1": "umap2d_y"})
    u3 = aligned(P / "umap3d_nn15_md0.1.parquet", ["c0", "c1", "c2"]).rename(
        {"c0": "umap3d_x", "c1": "umap3d_y", "c2": "umap3d_z"})
    sp = aligned(P / "umapsphere_nn15_md0.0_sp0.3.parquet", ["c0", "c1", "c2", "theta", "phi"]).rename(
        {"c0": "sphere_x", "c1": "sphere_y", "c2": "sphere_z", "theta": "sphere_theta", "phi": "sphere_phi"})
    col = aligned(P / "textcolor_d.parquet", ["r", "g", "b", "hue_deg", "agree"]).rename(
        {"r": "color_r", "g": "color_g", "b": "color_b", "hue_deg": "color_hue_deg", "agree": "color_agreement"})
    bas = aligned(Path("data/interim/placenames_basins.parquet"), ["basin_s0.3", "basin_s0.15"]).rename(
        {"basin_s0.3": "place_coarse_id", "basin_s0.15": "place_fine_id"})
    knn = aligned(P / "knn_dist.parquet", ["d1", "dk_mean"]).rename(
        {"d1": "nn1_cosine_distance", "dk_mean": "nn15_mean_cosine_distance"})

    meta = (
        corpus.join(kw, on="award_number", how="left").join(dai, on="award_number", how="left")
        .with_columns(
            ("https://kaken.nii.ac.jp/ja/grant/" + pl.col("kaken_id") + "/").alias("kaken_url"),
            pl.col("dai").alias("division"),
            pl.col("dai").replace_strict(DAI_GLOSS, default=None).alias("division_name"),
        )
        .select("award_number", "kaken_url", "title", "keywords", "category", "division", "division_name",
                "shokubun_codes", "start_fy", "end_fy", "n_tokens")
    )
    assert bool((meta["award_number"] == keys).all())
    meta = pl.concat([meta, u2, u3, sp, col, bas, knn], how="horizontal").with_columns(
        pl.col("place_coarse_id").replace_strict(name_of[0], default=None, return_dtype=pl.Utf8).alias("place_coarse"),
        pl.col("place_fine_id").replace_strict(name_of[1], default=None, return_dtype=pl.Utf8).alias("place_fine"),
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "embeddings").mkdir(parents=True)
    (OUT / "metadata").mkdir()
    (OUT / "extras").mkdir()
    meta.write_parquet(OUT / "metadata" / "train-00000-of-00001.parquet", compression="zstd")

    emb = pl.read_parquet(P / "embeddings.parquet")
    assert emb.height == n and bool((emb["award_number"] == keys).all())
    norms = np.linalg.norm(emb["embedding"].to_numpy()[:2000], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "埋め込みが L2 正規化されていない"
    step = (n + N_SHARDS - 1) // N_SHARDS
    for i in range(N_SHARDS):
        emb.slice(i * step, step).write_parquet(
            OUT / "embeddings" / f"train-{i:05d}-of-{N_SHARDS:05d}.parquet", compression="zstd")

    shutil.copy(P / "placenames_2d.json", OUT / "extras" / "placenames_2d.json")
    shutil.copy(P / "textcolor_legend.json", OUT / "extras" / "color_legend.json")
    shutil.copy("data/interim/elastic_ring_nodes.npz", OUT / "extras" / "elastic_ring_nodes.npz")

    card = CARD.read_text(encoding="utf-8").replace("{{N}}", f"{n:,}").replace("{{N_RAW}}", str(n))
    (OUT / "README.md").write_text(card, encoding="utf-8")

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{OUT}: {n:,} 件, 合計 {total / 1e6:.0f} MB")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  {f.stat().st_size / 1e6:.1f} MB")
    print(meta.head(2))
    print({c: str(t) for c, t in meta.schema.items()})
    print("null 件数:", {c: int(meta[c].null_count()) for c in meta.columns if meta[c].null_count()})


if __name__ == "__main__":
    main()
