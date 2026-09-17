---
license: cc-by-4.0
language:
- ja
pretty_name: KAKEN-ATLAS Embeddings
size_categories:
- 100K<n<1M
task_categories:
- feature-extraction
- sentence-similarity
tags:
- science-of-science
- scientometrics
- research-grants
- kakenhi
- embeddings
- umap
- map-of-science
configs:
- config_name: metadata
  data_files: metadata/*.parquet
  default: true
- config_name: embeddings
  data_files: embeddings/*.parquet
---

# KAKEN-ATLAS Embeddings

Text embeddings and map coordinates for **{{N}} Japanese research grants** (KAKENHI, Grants-in-Aid for Scientific Research,
projects starting in fiscal years 2019–2025). This is the data behind the interactive map of science
**[KAKEN-ATLAS](https://rma-lab.github.io/kaken-atlas/)** ([source code](https://github.com/rma-lab/kaken-atlas)).

科研費の採択課題 {{N}} 件（2019〜2025 年度開始）の**文章埋め込みと地図座標**です。インタラクティブな学術地図
[KAKEN-ATLAS](https://rma-lab.github.io/kaken-atlas/) の元データで、GPU がなくてもクラスタリングや近傍探索を追試できます。
日本語の説明は[後半](#日本語)にあります。

- Each project is embedded from its **title + keywords + abstract at the time of award** with
  [`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m) (768 dimensions, no prefix, L2-normalized).
- Positions and colors on the map are derived **from the text only**. Official review categories are not used for layout.
- **Not included**: abstracts (look them up on KAKEN by award number), researcher names, institutions, and budgets.
  The map is designed to be read by research content alone.

## Quick start

```python
from datasets import load_dataset
import numpy as np

meta = load_dataset("rma-lab/kaken-atlas-embeddings", "metadata", split="train").to_pandas()
emb = load_dataset("rma-lab/kaken-atlas-embeddings", "embeddings", split="train").with_format("numpy")
X = emb["embedding"]            # (N, 768), float32, L2-normalized; rows are aligned with `meta`

q = X[meta.index[meta.award_number == "23K20061"][0]]   # any award number in the data
top = np.argsort(-(X @ q))[:16]                          # cosine similarity = dot product
print(meta.loc[top, ["award_number", "title", "category"]])
```

Without the `datasets` library, read the parquet files directly (`polars.read_parquet`, `pandas.read_parquet`).
Row order is identical in every file, and `award_number` is a unique key.

## Files

| Path | Content |
|---|---|
| `metadata/*.parquet` | One row per project: attributes, map coordinates, color, place names (see columns below) |
| `embeddings/*.parquet` | `award_number`, `embedding` (float32 × 768, L2-normalized), 4 shards |
| `extras/placenames_2d.json` | Keyword place names of the 2D map: peak coordinates, sizes, candidate words (two levels) |
| `extras/color_legend.json` | 12 hue sectors of the content-derived color wheel with characteristic keywords |
| `extras/elastic_ring_nodes.npz` | The closed ring (198 nodes × 768 dims) fitted in the embedding space, and node hues |

### Columns of `metadata`

| Column | Meaning |
|---|---|
| `award_number` | KAKENHI award number (string, e.g. `23K20061`). Unique key |
| `kaken_url` | Project page on KAKEN (abstract, members, outputs) |
| `title`, `keywords` | Project title and author keywords (Japanese; some projects are in English) |
| `category` | Grant category (研究種目), e.g. 基盤研究(C), 若手研究 |
| `division`, `division_name` | Broad review division A–K derived from the official review-section table. `複数` = spans several divisions, `区分なし` = outside the scheme (e.g. JSPS fellows) |
| `shokubun_codes` | Official review-section codes (5-digit strings; keep the leading zeros) |
| `start_fy`, `end_fy` | Japanese fiscal years |
| `n_tokens` | Length of the embedded text in Ruri v3 tokens (median 168, max 661) |
| `umap2d_x`, `umap2d_y` | 2D UMAP (cosine, n_neighbors=15, min_dist=0.1, seed=42) |
| `umap3d_x/y/z` | 3D UMAP, same settings |
| `sphere_x/y/z`, `sphere_theta`, `sphere_phi` | Spherical UMAP (`output_metric="haversine"`, min_dist=0, spread=0.3); xyz on the unit sphere |
| `color_r/g/b` | Content-derived color used on the map (sRGB) |
| `color_hue_deg`, `color_agreement` | Position on the fitted ring (hue angle) and how consistent the neighboring ring nodes are (0–1; low = between fields) |
| `place_coarse_id`, `place_coarse` | Region of the 2D density landscape (42 regions) and its two-keyword name |
| `place_fine_id`, `place_fine` | Finer regions (129); `null` name = region too small to be named, id 0 = unassigned |
| `nn1_cosine_distance`, `nn15_mean_cosine_distance` | Cosine distance to the nearest / mean of the 15 nearest projects in the 768-d space |

## How it was made

1. **Data**: 238,997 records (FY2018–2025) fetched from the KAKEN OpenSearch API. The corpus keeps projects starting in
   FY2019–2025, excludes declined ones, and requires the abstract written at the time of award ({{N}} projects).
   FY2018 is excluded because only outcome abstracts exist for it; mixing proposal text and report text would bias the space.
2. **Embedding**: `cl-nagoya/ruri-v3-310m` (ModernBERT-Ja), empty prefix (semantic encoding), L2 normalization.
   Computed on one NVIDIA A40 of the JAIST HAKUSAN cluster in 29 minutes.
3. **Dimensionality reduction**: UMAP on cosine distance, fixed seed. The spherical version optimizes great-circle distance
   directly (it is not a flat map wrapped on a sphere).
4. **Color**: a closed ring (ring-shaped self-organizing map, 198 nodes) is fitted in the 768-d space; hue = position on the ring,
   saturation = agreement of neighboring nodes. Mean cosine between a project and its nearest node is 0.886.
5. **Place names**: peaks of the smoothed 2D density; regions by watershed on the inverted density; two keywords per region chosen
   by concentration `n_in / (n_all + 30) × n_in^0.5`, weighting projects near the peak more.

Details: [method notes](https://github.com/rma-lab/kaken-atlas/blob/master/doc/method.md) (Japanese) and the
[two-page flyer](https://github.com/rma-lab/kaken-atlas/blob/master/doc/flyer/kaken-atlas_flyer_2026-09.pdf).
The build script for this dataset is `scripts/build_hf_dataset.py` in the repository.

## Things to know before you use it

- **Do not interpret absolute cosine similarity.** All pairs fall in a narrow band (median 0.72, 5–95% = 0.68–0.78) because of
  the anisotropy of transformer embeddings. Use ranks or relative values.
- **UMAP preserves local neighborhoods only.** Distances between far-apart regions, areas, and the shape of empty space carry no
  meaning. Only about 10% of the 15 nearest neighbors in 768-d remain among the 15 nearest in 2D. **Measure closeness in 768-d.**
- An award-time abstract is a summary of a proposal, not of results. A small share of projects is written in English only, and
  their placement is less reliable because the model targets Japanese.
- The official categories (`division`, `shokubun_codes`) are provided for comparison. They were not used to build the space.

## License and attribution

The data are derived from the KAKEN database and may be used under **CC BY 4.0**, following the
[KAKEN terms of use](https://support.nii.ac.jp/kaken/about/terms). When you use them, please

- credit the source: **「KAKEN：科学研究費助成事業データベース（国立情報学研究所）」**
  (KAKEN: Grants-in-Aid for Scientific Research Database, National Institute of Informatics),
- state that the data were edited and processed by KAKEN-ATLAS (and by you, if you process them further), and
- do not present processed data as if the National Institute of Informatics had produced them.

Suggested credit: *Source: KAKEN database (National Institute of Informatics), edited and processed by KAKEN-ATLAS (JSPS KAKENHI 26K15524).*

The embedding model Ruri v3 is Apache-2.0. The code of KAKEN-ATLAS is MIT.

## Citation

```bibtex
@misc{kaken_atlas_embeddings_2026,
  author       = {Ogi, Takayuki},
  title        = {KAKEN-ATLAS Embeddings: text embeddings and map coordinates of Japanese research grants (FY2019--2025)},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/rma-lab/kaken-atlas-embeddings}},
  note         = {Map: https://rma-lab.github.io/kaken-atlas/ , code: https://github.com/rma-lab/kaken-atlas}
}
```

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 26K15524. The embeddings were computed on the HAKUSAN cluster of the
Japan Advanced Institute of Science and Technology (JAIST). Comments and corrections are welcome at
[GitHub Issues](https://github.com/rma-lab/kaken-atlas/issues).

---

## 日本語

### これは何か

科研費の採択課題 {{N}} 件（2019〜2025 年度開始、不採択を除く、採択時の研究概要があるもの）について、
**タイトル＋キーワード＋採択時概要**を日本語埋め込みモデル Ruri v3（`cl-nagoya/ruri-v3-310m`、768 次元、接頭辞なし、L2 正規化）で
ベクトルにしたものと、そこから作った地図の座標・色・地名です。学術地図 [KAKEN-ATLAS](https://rma-lab.github.io/kaken-atlas/) の元データです。

**含めていないもの**: 研究概要の本文（課題番号から KAKEN で参照できます）、研究者名、所属機関、配分額。
名前や機関の先入観なしに研究内容で学術を俯瞰する、という地図の方針に合わせています。

### ファイル

- `metadata/`: 1 課題 1 行。課題番号、KAKEN の URL、タイトル、キーワード、種目、大区分、小区分コード、年度、2D・3D・球面の座標、
  点の色（色相と合意度）、キーワード地名の山域（42 と 129 の 2 階層）、768 次元での最近傍距離。
- `embeddings/`: 課題番号と 768 次元ベクトル（float32、L2 正規化済み）。4 分割。行順は `metadata` と同じです。
- `extras/`: 地名の JSON、色の凡例、色の輪のノード（198 × 768）。

### 使う前に

- **コサイン類似度の絶対値は解釈しないでください。** 全ペアが 0.72 付近の狭い帯に集中します（埋め込みの異方性）。順位や相対値で使います。
- **UMAP が保つのは局所の近傍関係だけです。** 離れた領域どうしの距離、面積、空白の形に意味はありません。近さは 768 次元で測ってください。
- 採択時概要は申請書の要約で、成果とはずれます。英語のみの課題は配置の妥当性が下がります。
- 公式の区分（大区分、小区分コード）は比較のために添えたもので、空間を作るのには使っていません。

### 利用条件

KAKEN の[利用規程](https://support.nii.ac.jp/kaken/about/terms)（CC BY 4.0 と互換）に従い、次を守ってご利用ください。

- 出典の明記: **「KAKEN：科学研究費助成事業データベース（国立情報学研究所）」**
- 編集・加工したものである旨の明記（本データは KAKEN のデータを KAKEN-ATLAS が編集・加工したものです）
- 国立情報学研究所が作成したかのような態様で公表・利用しないこと

記載例: 出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを KAKEN-ATLAS（科研費 26K15524）が編集・加工

### 謝辞

本研究は JSPS 科研費 基盤研究(C) 26K15524 の助成を受けています。埋め込みの計算には北陸先端科学技術大学院大学（JAIST）の
計算機 HAKUSAN を利用しました。ご意見や誤りの指摘は [GitHub の Issues](https://github.com/rma-lab/kaken-atlas/issues) へお寄せください。
