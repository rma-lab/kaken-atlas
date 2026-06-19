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

## HAKUSAN 操作ルール（重要）
1. 操作前に到達性を確認する（VPN＝F5 が必要）。落ちていたら作業を止め、ユーザに再接続を依頼する。
2. ログインノード（hakusan1/2）では軽い読み取りコマンドのみ。計算は必ず SLURM 経由。
3. 実資源を消費するジョブ（GPU／長時間 CPU）は、区分・`--time`・リソース量を提示し**確認後に投入**。テストは短い `--time` で。
4. `salloc` を放置しない（原則 `sbatch`）。`rm`・上書き・`scancel`・クラスタ側の設定変更などの不可逆操作は事前確認。
- 既定の接続は `ssh hakusan`（= hakusan2）。

## 関連プロジェクト
- `~/Projects/research-latent-structure` … 本プロジェクトの前身（再利用可能な実装計画 `docs/PLAN.md` あり）。
- `~/Projects/kaken-summary` … KAKEN XML パースの先行知見（eradCode 同定・declined 除外など）。
