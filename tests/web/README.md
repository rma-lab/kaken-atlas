# 地図サイトの回帰試験（ヘッドレス Chrome）

`docs/` をローカル HTTP で配信し、puppeteer-core で実物のページを操作して確認する。UI を変えたら公開前に全部通す。

```bash
cd tests/web
npm install            # 初回のみ（puppeteer-core。Chrome 本体は手元の Google Chrome を使う）
node run.js            # 全部（3〜5 分）
node run.js lasso perf # 名前の一部で絞る
```

- Chrome の場所は `CHROME_PATH`、対象の docs は `KA_DOCS`（既定はリポジトリの docs/）、描画は既定で Metal の GPU
  （`KA_GL=swiftshader` で切替。深度・時間の検証は GPU でないと当てにならない）。
- スクリーンショットは `out/`（git 管理外）。
- 各テストは `tests/NN_名前.js`。`async (ctx) => [{name, ok, info}]` を返す。共通部品は `lib.js`。

| テスト | 見るもの |
|---|---|
| 01_data | 3 ビューの読み込み、シャード行と種目の整合、点ごとの色、トレース数 ≤ 255 |
| 02_deeplink | `?award=` で 3 ビューにカードとリング、存在しない番号 |
| 03_search_card | 検索ヒット→カード（スマホは結果を閉じてカメラ移動、PC は縁・見出し＝点の色） |
| 04_aspect | 2D の縦横比（初期・歪んだ範囲・フライトゥ・リサイズ・ダブルクリック） |
| 05_lasso | 自前なげなわ（速さ、直後の移動、囲いの追随、膜の上の操作、閉じる／Esc） |
| 06_offline | Service Worker（2 回目はキャッシュ、オフラインで開いて検索） |
| 07_perf | 2D 範囲変更の中央値 < 80ms（scaleanchor の罠の再発防止） |
| 08_panels | 色の見方（PC・スマホ）、大区分シート |
| 09_stale_cache | 旧版（v1.2）を保存後に新版へ切替、1 回目で正常（データ URL の版） |

## リファクタリングの安全網（動作を変えていないことの確認）

```bash
uv run python tests/web/artifacts.py check   # 生成物の同一性: points.bin×3、shards×101、manifest×3、sw.js のデータ版
uv run python tests/web/artifacts.py record  # 基準を baseline/artifacts.json に記録（git 管理）。データを意図して更新したときだけ
node shots.js record                          # 基準スクリーンショットを out/baseline/ に撮る（git 管理外。整理を始める前に）
node shots.js check                           # 撮り直して画素差分（差分画素 0.3% 超で NG。差分画像は out/diff/）
```

- `artifacts.py` は Python 側（build_order、manifest、シャード）の整理用。整理の前後で 14 項目すべて一致すること。
- `shots.js` は 9 構図（3 ビュー × PC・スマホ、2D のフライトゥ＋カード、色の見方、なげなわ）。同じ機械・GPU なら差分 0.000% で
  再現する（確認済み）。UI を意図して変えたときは基準を撮り直す。

時間依存の検証（慣性など）はページ内で合成 TouchEvent を発行する必要があり、ここには含めていない（`doc/web.md` §8）。
