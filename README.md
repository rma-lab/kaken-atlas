# KAKEN-ATLAS

**KAKEN Atlas of Latent Academic Structures**
科研費データを用いた学術知識の構造化と研究評価の新指標開発

JSPS 科研費 基盤研究(C) 26K15524（令和8〜10年度 / FY2026–2028）。
日本最大規模の学術データベースである科研費（KAKEN）データに教師なし深層学習を適用し、
既存の306小区分に依存しない学術研究の潜在構造を解明する。さらに埋め込み空間での
位置から **新規性・架橋性・成長性** の3指標を開発し、引用を待たずに研究価値を評価する。

## 🗺️ 学術地図（公開中）

**https://rma-lab.github.io/kaken-atlas/**

2019–2025年度開始の採択課題 206,078件を意味空間上に配置したインタラクティブ地図
（2D/3D）。タイトル・キーワード検索、種目フィルタ、なげなわ選択による集計、
点クリックでKAKEN課題ページへのジャンプができる。実体はこのリポジトリの
`docs/` を GitHub Pages で配信したもの（`scripts/build_web_map.py` で生成）。

## 研究計画（年次）

| 年度 | テーマ | 状況 |
|------|--------|------|
| **R8 (2026)** | データ基盤構築・埋め込み空間生成 | 取得238,997件 → コーパス206,078件 → 768次元埋め込み → UMAP 2D/3D地図・年度別KDE時系列（**完了**） |
| **R9 (2027)** | 潜在構造の発見・新指標開発 | UMAP→HDBSCAN クラスタリング → 既存306小区分との乖離分析 → 3指標開発 |
| **R10 (2028)** | 実証実験・システム実証 | 全国URAによる評価実験 → Webアプリ → 論文投稿 |

## パイプライン

```bash
uv sync                                  # 環境構築（uv.lock で再現）
uv run python -m kaken_atlas.fetch       # KAKEN opensearch API から取得（要 .env の KAKEN_APP_ID）
uv run python -m kaken_atlas.parse       # XML → data/interim/awards.parquet
uv run python -m kaken_atlas.corpus      # 埋め込み対象コーパス → data/processed/corpus.parquet
uv run python -m kaken_atlas.embed       # Ruri v3 で768次元埋め込み（GPU推奨、A40で約30分）
uv run python -m kaken_atlas.reduce      # UMAP 2D（--n-components 3 で3D）
uv run python scripts/build_web_map.py data/processed/umap2d_nn15_md0.1.parquet  # 地図サイト生成
```

- データ取得には KAKEN の appid（[国立情報学研究所に利用申請](https://support.nii.ac.jp/ja/cinii/api/developer)）
  が必要。`.env` に `KAKEN_APP_ID=...` として置く（git 管理外）。
- 生データ・中間生成物（`data/`）は git 管理外。上のコマンドで再構築できる。

## 技術スタック

- **言語 / 環境**: Python 3.12（[uv](https://docs.astral.sh/uv/) で管理）
- **埋め込み**: [Ruri v3 310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)（768次元、ModernBERT-Ja ベース、Apache-2.0）
- **次元削減**: UMAP（cosine、seed 固定）
- **クラスタリング（R9予定）**: HDBSCAN
- **可視化**: matplotlib（静的図・時系列KDE）/ Plotly（インタラクティブ地図）
- **計算環境**: 開発=ローカル（CPU/MPS で可）/ 埋め込みは GPU 1枚で約30分（NVIDIA A40 実測）

## ディレクトリ構成

```
kaken-atlas/
├── src/kaken_atlas/   # パイプライン本体（fetch / parse / corpus / embed / reduce / kubun）
├── scripts/           # 可視化・地図サイト生成・審査区分マスタ構築
├── docs/              # GitHub Pages（公開地図サイト）＋設計メモ
├── data/
│   ├── reference/     # 審査区分マスタ（正典、git 管理）
│   ├── raw/           # KAKEN API 生データ（git 管理外）
│   ├── interim/       # 前処理途中（git 管理外）
│   └── processed/     # コーパス・埋め込み・UMAP座標（git 管理外）
└── tests/
```

## データ出典

科研費データは [KAKEN科研費データベース](https://kaken.nii.ac.jp/)（国立情報学研究所）より
取得・加工。地図・図表を利用する際は KAKEN の出典を明記してください。

## ライセンス

MIT
