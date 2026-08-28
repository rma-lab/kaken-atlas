# kubun_table — 科研費審査区分表（機械可読版）

2026-08-28 生成。生成: `uv run python scripts/build_kubun_table.py`（既定出力先=このディレクトリ）。

## 出所（provenance）

- **階層・コード・名称（日英）**: NII 公式マスタ
  [niijp/grants_masterxml_kaken](https://bitbucket.org/niijp/grants_masterxml_kaken)
  の `review_section_master_kakenhi.xml`（type="review_section", 2018-04-01 施行の審査区分表）
- **小区分の内容キーワード**: 手作りの `KubunTable.csv`（`~/Library/CloudStorage/Dropbox/研究IR/区分分類/BERT/`、Shift-JIS）。
  2026-08-28 に公式マスタおよび実データ（KAKEN 採択課題23.9万件）と全数照合し、
  コード・名称・所属関係すべて正しいことを確認済み

## ファイル

### kubun_table.csv（UTF-8、tidy 形式、323行）

**1行 = 小区分の1所属**。複数の中区分に属する小区分（14種）は複数行になる。

| 列 | 内容 |
|---|---|
| dai_code | 大区分 A〜K |
| chu_code / chu_name_ja / chu_name_en | 中区分 1〜65 と名称 |
| sho_code | 小区分コード（5桁ゼロ埋め文字列。例 "01010"） |
| sho_name_ja / sho_name_en | 小区分名称 |
| sho_keywords | 小区分の内容（審査区分表の「内容」欄） |
| n_memberships | この小区分が持つ所属数（1 なら単一所属、2〜3 は複数所属） |

### kubun_table.json（UTF-8）

小区分コードをキーにした辞書。LLM やプログラムに渡す用。

```json
"90020": {
  "name_ja": "図書館情報学および人文社会情報学関連",
  "name_en": "...",
  "keywords": "図書館学、情報サービス、…",
  "memberships": [
    {"dai_code": "A", "chu_code": 2, "chu_name_ja": "文学、言語学およびその関連分野"},
    {"dai_code": "J", "chu_code": 62, "chu_name_ja": "応用情報学およびその関連分野"}
  ]
}
```

## 注意

- 大区分に公式の主題名はない（A〜K のみ）
- M付きの合同審査区分（例 M01010-01080）は含まない（実データの審査区分に出現しないため）
- 小区分コードは必ず**5桁ゼロ埋めの文字列**として扱うこと（数値にすると先頭ゼロが落ちる）
- 複数所属の小区分を大区分別に集計・彩色する場合は、扱い（主所属/学際扱い/按分）を明示的に決めること
