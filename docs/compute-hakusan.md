# 計算環境メモ：JAIST HAKUSAN

KAKEN-ATLAS の大規模 BERT 埋め込み（R8）を実行する GPU 環境として **JAIST HAKUSAN** を用いる。
本メモは実機ログイン（2026-06）で確認した内容と、公式講習会資料
（HPCソリューションズ, 2026-04-14, `~/Library/CloudStorage/Dropbox/JAIST/HAKUSAN/`）に基づく。
※ 講習会資料は一部が旧テンプレ値だったため、**実機 `spart`/`module avail` の値を正**とする。

---

## 1. アクセス

```bash
ssh hakusan      # ~/.ssh/config のエイリアス → hakusan2.jaist.ac.jp (user: takayuki)
ssh hakusan1     # 混雑時の予備
```

- **既定は hakusan2**（hakusan1 は混みやすい）。
- 鍵認証（`~/.ssh/id_ed25519`、HAKUSAN 側 `authorized_keys` に登録済み、パスフレーズ無し）。
  Claude Code から非対話で接続可能。`~/.ssh/config` に ControlMaster（接続再利用）設定あり。
- **ログインノード（hakusan1/2）で重い処理を走らせない。** 計算は必ず SLURM 経由。

## 2. システム構成（実機）

| 種別 | ノード | 仕様 | OS |
|------|--------|------|-----|
| ログイン | hakusan1, hakusan2 | 8 CPU / 128GiB（JAISTクラウド上のVM） | Ubuntu 24.04 |
| 計算 | lcpcc-001〜124 | Intel Xeon 6980P, 256core, 1.5TiB | Ubuntu 24.04 |
| GPU (A100) | spcc-a100g01〜10 | NVIDIA **A100 40GB** ×2/node, 52core, 512GiB | Ubuntu 20.04 |
| GPU (A40) | spcc-a40g01〜20 | NVIDIA **A40 48GB** ×2/node, 52core, 512GiB | Ubuntu 20.04 |
| クラウドGPU | spcc-cld-gl01〜04 | NVIDIA **H100 80GB**（分割） | Ubuntu 22.04 |

ストレージ：
- `/home/takayuki` … JAIST 全学共通 NFS。**quota 無制限**（現状の確認では制限なし）、巨大（PB級）。
- `/JOBs`（`$JOBDIR`）… Lustre 高速スクラッチ（〜236TB）。ジョブごとに作成され**ジョブ終了時に削除**。
  I/O の多い処理はここにステージング（実行前にコピー → 実行後に `/home` へ戻す）。

## 3. ソフトウェア環境

- **CUDA**: `module load cuda/13.3`（ほか NVIDIA HPC SDK `nvhpc/26.3`、cuda12/cuda13 版あり）。
- **Python / conda / apptainer のモジュールは無い** → 自前で用意する。
  - システム Python は `/usr/bin/python3` = **3.12.3**（偶然 KAKEN-ATLAS のピンと一致）。
  - 本プロジェクトは **uv** で環境を持ち込む（root 不要、`~/.local/bin` に導入）。
- **PyTorch の CUDA ランタイムはホイール同梱**なので、ノードに NVIDIA ドライバがあれば動く。
  → `module load cuda` は必須ではない。CUDA 13.3 世代のため torch の cu12x/cu13x ホイールが利用可。
  ドライバ実バージョンは GPU ノードで `nvidia-smi` 確認（cu128 / cu130 の選定用）。

### uv の導入（初回のみ、HAKUSAN 上で）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ~/.local/bin が PATH に入る。新しいシェルで uv が使える。
```

## 4. SLURM（バッチスケジューラ）

主なコマンド：

```bash
sbatch job.sh             # バッチ投入
salloc -p <PARTITION>     # 対話確保（使い終わったら必ず exit で解放）
squeue -u $USER           # 自分のジョブ確認
sacct -j <JOBID> --format=JobID,Partition,MaxRSS,Elapsed,State   # 実績
scancel <JOBID>           # 取消
spart                     # 区分一覧（JAIST 独自）
sinfo -p <PARTITION> -o "%n %t %G"   # ノード空き状況
```

### 主な PARTITION（実機 `spart`）

CPU 系（lcpcc ノード）：

| Partition | 上限時間 | ノード | 備考 |
|-----------|---------|--------|------|
| DEF*（既定） | 7日 | 1 | cpu≤64, mem≤384G |
| TINY | 30分 | 1 | 試験用 |
| SINGLE | 7日 | 1 | cpu≤256, mem≤1.5T |
| SMALL/LARGE/XLARGE/X2LARGE | 5〜7日 | 1〜32 | マルチノード(要MPI) |
| LONG / LONG-L | 14〜21日 | 1〜3 | 長時間 |

GPU 系：

| Partition | GPU | 構成 | 上限 | 同時実行/人 |
|-----------|-----|------|------|-------------|
| **GPU-1** | A40 ×1 | 26core/256G | 7日 | 4 |
| GPU-S | A40 ×2 | 52core/512G | 5日 | 2 |
| GPU-L | A40 ×8 | 208core/2T | 3日 | 1 |
| GPU-1A | A100 ×1 | 26core/256G | 7日 | 2 |
| GPU-LA | A100 ×8 | 208core/2T | 3日 | 1 |
| VM-GPU-L | H100 80GB ×1 | 32core/480G | 2日 | 1 |
| SEMINAR | RTX PRO 6000 Blackwell 24GB | — | 3.5日 | — |

> **GPU は計算ノードでのみ利用可（ログインノード不可）。** GPU 要求には `--gres=gpu:1` を付ける。

### バッチジョブ雛形

```bash
#!/bin/bash
#SBATCH -p GPU-1                 # 区分
#SBATCH --gres=gpu:1             # GPU 1枚
#SBATCH --time=00:30:00          # 上限時間（短いほどバックフィルで早く回る）
#SBATCH -J atlas-embed           # ジョブ名
#SBATCH -o logs/%x-%j.out        # 標準出力（%x=ジョブ名, %j=JOBID）
#SBATCH -e logs/%x-%j.out
#SBATCH --mail-type=END,FAIL     # 完了/失敗を通知
#SBATCH --mail-user=takayuki@jaist.ac.jp

cd ${SLURM_SUBMIT_DIR}
source ~/.local/bin/env          # uv を PATH に
uv run python scripts/embed.py   # 実行（プロジェクトの venv で）
```

投入と監視：

```bash
sbatch job.sh
squeue -u $USER -o "%.10i %.8P %.10j %.2t %.10M %.20R"
```

## 5. 運用上の知見・方針（KAKEN-ATLAS 向け）

- **GPU 区分は混雑することがある**（実測で GPU-1 に多数の PD ジョブ）。
  即時の対話確保（`srun --immediate`）は通りにくい。**短時間バッチ＋バックフィル**を基本にする。
- 本番の埋め込み（BERT-base, 約1.1億パラメータ, 約10万件の**推論のみ**）は GPU 1枚で十分。
  10万件を**チャンク分割して複数ジョブ（GPU-1 は同時4本/人）**に流すと、混雑下でも総時間を短縮できる。
- 大きめの試行は A100（GPU-1A）や H100（VM-GPU-L）も選択可。
- 結果（埋め込みベクトル等）は `/home` に保存。`$JOBDIR` はジョブ終了で消えるので戻し忘れに注意。
- すべての操作は Claude Code 経由でログが残る＝**再現性の確保**にも資する（本研究の主旨と整合）。

## 6. 安全ルール（Claude Code から操作する際）

1. ログインノードでは読み取り系の軽いコマンドのみ。計算は SLURM 経由。
2. 実資源を消費するジョブ（GPU / 長時間 CPU）は、区分・`--time`・リソース量を提示し**確認後に投入**。試験は短い `--time` で。
3. `salloc` は開けっ放しにせず即解放。原則 `sbatch`。
4. 取り返しのつかない操作（`rm`・上書き・他ジョブの `scancel`・クラスタ側の鍵/設定変更）は事前確認。

## 7. 未確定・TODO

- [ ] GPU ノードでの `nvidia-smi`（ドライバ実バージョン → torch cu128/cu130 の確定）
- [ ] HAKUSAN 上に uv を導入し、プロジェクトの venv を構築（Python 3.12）
- [ ] torch（CUDA 版）の動作確認ジョブ（`torch.cuda.is_available()`）

## 参考リンク

- SLURM ユーザガイド: https://slurm.schedmd.com/
- NVIDIA CUDA: https://docs.nvidia.com/cuda/
- Apptainer/Singularity: https://docs.sylabs.io/guides/4.3/user-guide/
