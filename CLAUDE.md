# KAKEN-ATLAS — プロジェクト指示

**KAKEN Atlas of Latent Academic Structures**。科研費データに教師なし深層学習を適用し、
既存306小区分に依存しない学術研究の潜在構造を解明し、**新規性・架橋性・成長性**の3指標を開発する。
（JSPS 科研費 基盤研究(C)、令和8〜10年度 / FY2026–2028）

> このファイルは「守るべき規約・確定事項」を記す。**決定の経緯・調査ログ・個人的文脈**は
> Claude のメモリ（`~/.claude/projects/-Users-takayuki-Projects-kaken-atlas/memory/`）側にある。

## 環境・ツール
- パッケージ／環境管理は **uv**（`pip`/`venv` を直接使わない）。`uv add` / `uv sync` / `uv run` を使う。
- Python は **3.12** にピン（`.python-version`）。理由：GPUクラスタのCUDAホイール安定性。**3.14は使わない。**
- `uv.lock` はコミット、`.venv/` は git 除外。

## 主要な技術決定
- **埋め込みモデル：Ruri v3 310m**（`cl-nagoya/ruri-v3-310m`、768次元、Apache-2.0、最大8192トークン）。
  ModernBERT-Ja ベース。クラスタリング用途では **空プレフィックス（意味エンコード）モード**を使う。
  ※申請書記載の `cl-tohoku/bert-base-japanese-v3` ではなく、性能優先でRuriを採用（申請書の方針より優先）。
- **データ源：KAKEN opensearch API**（`https://kaken.nii.ac.jp/opensearch/`、`format=xml`、`rw=500`、`st` ページング）。
  **appid 必須** → `.env` の `KAKEN_APP_ID`（git 除外）に置く。コードに直書きしない。
- **データ対象範囲：2018年度以降のみ**（小区分306は2018年の審査区分改革で導入。それ以前は比較対象外）。
- **計算環境**：開発＝ローカル Mac（CPU/MPS）、大規模埋め込み＝JAIST HAKUSAN GPU。詳細は `docs/compute-hakusan.md`。

## ディレクトリ・データ規約
- `data/raw/` ＝ 取得した生データ（不変・git 除外）、`data/interim/` ＝ 前処理途中、`data/processed/` ＝ 成果物。
- 秘密情報（appid 等）は `.env`（git 除外）。

## データパイプライン（取得・パース済み：2026-07-17）
- 取得：`uv run python -m kaken_atlas.fetch`（2018–2025年度・238,997件 → `data/raw/opensearch/<年度>/`、計9.6GB）。
  取得済みページは自動スキップ（再実行安全）。**注意：`rw` は 20/50/100/200/500 のみ。仕様外の値は API がエラーを返さず黙って0件を返す。**
- パース：`uv run python -m kaken_atlas.parse` → `data/interim/awards.parquet`。
  パース時はフィルタしない（declined 除外・小区分絞り込み等は `status_code`・`shokubun_codes` 列で下流にて）。
- **採択時の研究概要（`abstract_initial`）は2019年度採択分から**しか存在しない。2018年度開始課題は
  成果概要（`abstract_achievement`）のみ → **埋め込みコーパスは2019年度以降に限定（2018年度は除外）**（2026-08決定）。
  2019年度以降は 209,734件中 206,906件（98.7%）に `abstract_initial` あり。
- コーパス構築：`uv run python -m kaken_atlas.corpus` → `data/processed/corpus.parquet`（206,078件）。
  フィルタ＝2019–2025年度・declined除外・`abstract_initial`あり。`text` 列＝タイトル＋キーワード＋概要の改行連結
  （Ruri v3 は空プレフィックスなので接頭辞なし）。`n_tokens`（Ruri実測）：中央値168・最大661 → 8192超なし。
- 埋め込み（済 2026-08-10）：`uv run python -m kaken_atlas.embed` → `data/processed/embeddings.parquet`
  （561MB、L2正規化済み768次元float32、行順は corpus.parquet と同一）。本番は HAKUSAN GPU-1/A40 で29分
  （`scripts/hakusan_embed.sh`）。**Linux の torch は cu128 ビルド固定**（GPUノードのドライバが CUDA 12.9 世代のため。
  `pyproject.toml` の `tool.uv.sources` 参照）。

## HAKUSAN 操作ルール（重要）
1. 操作前に到達性を確認する（VPN＝F5 が必要）。落ちていたら作業を止め、ユーザに再接続を依頼する。
2. ログインノード（hakusan1/2）では軽い読み取りコマンドのみ。計算は必ず SLURM 経由。
3. 実資源を消費するジョブ（GPU／長時間 CPU）は、区分・`--time`・リソース量を提示し**確認後に投入**。テストは短い `--time` で。
4. `salloc` を放置しない（原則 `sbatch`）。`rm`・上書き・`scancel`・クラスタ側の設定変更などの不可逆操作は事前確認。
- 既定の接続は `ssh hakusan`（= hakusan2）。

## 関連プロジェクト
- `~/Projects/research-latent-structure` … 本プロジェクトの前身（再利用可能な実装計画 `docs/PLAN.md` あり）。
- `~/Projects/kaken-summary` … KAKEN XML パースの先行知見（eradCode 同定・declined 除外など）。
