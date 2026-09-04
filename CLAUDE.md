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
- **計算環境**：開発＝ローカル Mac（CPU/MPS）、大規模埋め込み＝JAIST HAKUSAN GPU。詳細は `reports/compute-hakusan.md`
  （公開リポジトリから外すため git 管理外の reports/ に置く。2026-09-04）。

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

## 審査区分表・大区分の扱い
- **正典は `data/reference/kubun_table.csv` / `.json`**（公式マスタ由来、`scripts/build_kubun_table.py` で再生成、
  実データ23.9万件と全数照合済み）。小区分コードは**5桁ゼロ埋め文字列**で扱う（数値化禁止）。
- 大区分の割り当ては `kaken_atlas/kubun.py` の `load_dai_labels()` に一本化。審査区分の**3階層を解釈**：
  基盤B/C・若手=小区分、基盤A・挑戦的=中区分、基盤S=大区分直接。複数大区分にまたがる場合は「複数」、
  体系外（特別研究員奨励費・スタート支援・新学術・学術変革等）は「区分なし」。
- 大区分11色は**意味順色相環**（重心距離の最短巡回路→OKLCH等間隔。導出は `scripts/plot_map_dai.py` コメント）。

## 可視化パイプライン
- 次元削減：`uv run python -m kaken_atlas.reduce [--n-components 3]` → `data/processed/umap*_nn15_md0.1.parquet`
  （cosine・seed=42固定。2Dは M3 Air で約2分）。
- 静的図：`scripts/plot_map.py`（密度）、`scripts/plot_map_dai.py`（大区分パネル・一覧）→ `reports/figures/*.png`。
  **全図に出所・件数・年度・パラメータの脚注を焼き込む**こと。
- 時系列：`scripts/plot_kde_years.py` → 年度別KDEの図3枚（7パネル・差分マップ・生成AI分布）＋
  偏差動画2本（対全期間平均／対2019累積）。**設計原則**：①レイアウトは全期間一括UMAPで固定
  ②各年度は件数で正規化したシェア密度 ③動画は密度でなく偏差を描く（密度は年々ほぼ不変で動かない）
  ④偏差は**|偏差|>2σのみ着色**（σ=年度内半分割ブートストラップの画素別推定。σ=0.5でも年次揺らぎの
  大半が標本ノイズのため）⑤クロスフェードは演出であり中間状態の推定でない旨を脚注明記。
  差分・偏差の配色は**赤=増加/青=減少**で統一。横長図は左=地図・右=説明/脚注の分離レイアウト。
- インタラクティブ：`scripts/plot_map_interactive.py <座標parquet>` → 自己完結HTML（約135MB）。
  検索・種目フィルタ・なげなわ集計・KAKENリンク等はスクリプト内 POST_SCRIPT（JS）に実装。
- `reports/` は **git 管理外**（実験ノート扱い。公開物として文脈を整えるまでローカル）。
- UI 変更時は**ヘッドレスChromeで検証**（puppeteer-core、ハーネスは /tmp/ptest に作る流儀。
  小さい座標サブセットでテストHTMLを作り、マウス操作を再現して回帰確認してから渡す）。

## 発表資料
- Marp 原稿 `reports/slides/*.md` → `npx -y @marp-team/marp-cli <md> --pdf --allow-local-files` で PDF 化。

## HAKUSAN 操作ルール（重要）
1. 操作前に到達性を確認する（VPN＝F5 が必要）。落ちていたら作業を止め、ユーザに再接続を依頼する。
2. ログインノード（hakusan1/2）では軽い読み取りコマンドのみ。計算は必ず SLURM 経由。
3. 実資源を消費するジョブ（GPU／長時間 CPU）は、区分・`--time`・リソース量を提示し**確認後に投入**。テストは短い `--time` で。
4. `salloc` を放置しない（原則 `sbatch`）。`rm`・上書き・`scancel`・クラスタ側の設定変更などの不可逆操作は事前確認。
- 既定の接続は `ssh hakusan`（= hakusan2）。

## 関連プロジェクト
- `~/Projects/research-latent-structure` … 本プロジェクトの前身（再利用可能な実装計画 `docs/PLAN.md` あり）。
- `~/Projects/kaken-summary` … KAKEN XML パースの先行知見（eradCode 同定・declined 除外など）。
