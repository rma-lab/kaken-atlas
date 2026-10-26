"""kubun2028_merges.py の結果（merge_adjacency.md と図 8 枚）を A4 の PDF レポートにまとめる（2026-10-26）。

pandoc で md を HTML にし、前付（要旨・背景・方法・考察）と図を差し込んで、ヘッドレス Chrome で PDF に印刷する。
使い方: uv run python scripts/kubun2028_report.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MD = Path("reports/kubun2028/merge_adjacency.md")
FINDINGS = Path("reports/kubun2028/findings.md")  # 手書きの所見（merges.py の出力とは別に保守）
FIG = Path("reports/figures").resolve()
HTML = Path("reports/kubun2028/report_2026-10-26.html")
PDF = Path("reports/kubun2028/report_2026-10-26.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Hiragino Sans", "Hiragino Kaku Gothic ProN", sans-serif; color: #0b0b0b; font-size: 10.5pt; line-height: 1.65; }
h1 { font-size: 20pt; line-height: 1.35; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5px solid #0b0b0b; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; }
p { margin: 4pt 0 6pt; }
ul { margin: 2pt 0 6pt; padding-left: 1.4em; }
li { margin: 1pt 0; }
.meta { color: #898781; font-size: 9.5pt; margin-bottom: 14pt; }
.abstract { background: #f3f2ee; padding: 10pt 12pt; border-radius: 4pt; margin: 10pt 0 14pt; }
.abstract p { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #d9d8d3; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { border-bottom: 1.5px solid #0b0b0b; background: #faf9f6; }
table.codes td:first-child { white-space: nowrap; }
table.merge td:first-child { width: 30%; }
table.merge td:nth-child(2) { white-space: nowrap; }
table.pairs td, table.pairs th { text-align: center; font-size: 8pt; padding: 2pt 3pt; }
table.pairs td:first-child { text-align: left; white-space: nowrap; }
.verdict td:nth-child(1) { white-space: nowrap; font-weight: 600; }
.ok { color: #256abf; } .part { color: #8a6d1f; } .ng { color: #b3261e; } .na { color: #898781; }
figure { margin: 8pt 0 6pt; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 9pt; color: #898781; margin-top: 2pt; }
.section { page-break-before: always; }
.foot { color: #898781; font-size: 8.5pt; margin-top: 16pt; border-top: 1px solid #d9d8d3; padding-top: 6pt; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f3f2ee; padding: 0 3pt; }
"""

FRONT = """
<h1>令和 10（2028）年度 審査区分表の改正で統合・吸収される小区分は、研究内容の上で隣だったか</h1>
<p class="meta">KAKEN-ATLAS（JSPS 科研費 26K15524）検証ノート　2026 年 10 月 26 日　対象: 2019–2025 年度開始の採択課題のうち審査区分が 1 つの 169,500 件</p>

<div class="abstract">
<p><b>要旨</b>　文部科学省は 2026 年 7 月 29 日、令和 10 年度公募から適用する科研費「審査区分表」の改正を決定した（大区分 11→13、小区分 306→296）。
本ノートは、削除・統合される 13 小区分について、採択課題の研究概要の埋め込み（Ruri v3、768 次元）を用い、「制度が統合した相手は、同じ中区分の他の区分と比べても内容上もっとも近かったか」「両者の境界は曖昧だったか」を検証した。
結果、13 件のうち<b>内容と整合するものが 5 件</b>（社会法＋新領域法、生物資源保全→自然共生システム、認知脳科学→基盤脳科学、英文学＋ヨーロッパ文学、医療福祉工学→リハビリテーション科学）、
<b>部分的に整合が 3 件</b>（中国文学、数学基礎、国際法。中区分内では近いが内容上の最近傍は別の中区分にある）、
<b>内容と整合しないものが 4 件</b>（基礎法学＋刑事法学、理論経済学＋経済学説、日本語学＋英語学、医療技術評価→医療管理学）、判定不能が 1 件（神経系。旧区分が連続体で、新表は別の軸で切る）であった。
統合の多くは「中区分を維持する」という制度上の制約の中で相手を選んでおり、内容上の最近傍が中区分の外にある例が複数見られる。唯一の大区分をまたぐ吸収（生物資源保全学→環境）は、データ上も最も自然な移動だった。</p>
</div>

<h2>1. 背景</h2>
<p>科研費の審査区分表は 2018 年度の審査システム改革で導入され、大区分（A〜K）・中区分・小区分（306）の 3 層からなる。令和 10 年度公募から適用される今回の改正の骨子は次のとおり（文科省 科学研究費補助金審査部会「科学研究費助成事業『審査区分表』の改正等について」2026-07-29）。</p>
<ul>
<li>大区分 11 → 13（A〜M）。応募件数が多く小委員会が 3 分割していた旧 B・D を再編。新 C は素粒子・原子核・宇宙物理と天文学、旧 C・D の工学は新 D・E・F に三分割。</li>
<li>中区分 91「量子情報およびその関連分野」を新設し、B・C・E・F・L の 5 大区分に分属。</li>
<li>小区分は 13 削除・3 新設で 306 → 296。5 桁コードが振り直され、同じ番号が別の意味で再利用される（例 02020: 中国文学 → 英語圏文学およびヨーロッパ文学）。</li>
<li>基盤研究(B) の合同審査区分を解消。内容の例は 450 語追加・383 語削除。</li>
</ul>
<p>見直しの基本方針は「区分の数は増やさない」「応募件数が少ない区分は統合を積極的に検討する」「中区分は維持を基本とする」であり、統合の判断は主として応募件数と審査運営の観点からなされている。本ノートの問いは、その統合が<b>研究内容の近さ</b>とどの程度一致していたかである。</p>

<h2>2. データと方法</h2>
<p><b>データ</b>　KAKEN の採択課題 2019–2025 年度開始分のうち、採択時の研究概要があり、審査区分（小区分）が 1 つだけのもの 169,500 件。各課題はタイトル・キーワード・概要を連結した本文を cl-nagoya/ruri-v3-310m で埋め込んだ 768 次元ベクトルで表す（KAKEN-ATLAS の本番埋め込み）。全体平均を引いて異方性を除いたうえで cosine 距離を用いる。地図（UMAP）の座標は計算には使わず、図示にのみ使う。</p>
<p><b>指標</b>　削除・統合される小区分（統合元）と制度上の行き先（相手）について次を求めた。</p>
<ul>
<li><b>重心順位</b>: 統合元の重心から見て、同じ中区分の小区分のうち相手が何番目に近いか（1 が最近傍）。括弧内は全 306 小区分での順位。</li>
<li><b>近傍リフト</b>: 統合元の各課題の近傍 30 件（自区分を除く）に占める相手区分の割合を、相手区分の全体シェアで割った値。1 なら無関係。相手が小さい区分ほど大きく出るため補助指標。</li>
<li><b>最多近傍</b>: 統合元の課題の近傍（自区分を除く）で最も多い小区分の上位 3。制度の行き先と一致するか。</li>
<li><b>分類精度</b>: 統合元と相手を PCA 50 次元のロジスティック回帰で分ける 5 分割交差検証の balanced accuracy。0.5 が区別不能。同じ中区分の全ペアで同じ計算をし、その中での順位（1 が最も区別しにくい）を添える。</li>
<li><b>按分</b>（分割吸収の場合）: 統合元の各課題について、近傍 30 件に二つの行き先のどちらが多いかの比率。</li>
</ul>
<p><b>判定基準</b>　①相手が中区分内で最も近い区分（重心順位 1 または最多近傍）であり、②分類精度が中区分内で低い側にあるとき「整合」。①が中区分内では成り立つが内容上の最近傍が別の中区分にあるとき「部分的」。相手が中区分内で遠い側、または最も混ざる相手が別にあるとき「不整合」。</p>
<p><b>図の読み方</b>　各中区分の図は左右一対。左は全体地図（2D UMAP、20.6 万件）の該当領域の拡大で、灰色は他の課題。右はその中区分の課題だけで UMAP を計算し直したもので、内部で二つの区分が混ざっているか住み分けているかを見る。UMAP は塊の分離を誇張しやすいので、絵の印象は必ず数表と突き合わせる。★は統合・吸収に関わる区分。</p>
<p><b>限界</b>　小区分は応募者の自己申告であり、内容の近さと一致しないことがある。小区分どうしは一般に分類精度 0.8 以上で分かれる（埋め込みが分野語彙を拾う）ため、「曖昧」は相対評価である。基盤研究(B) の 2023 年度以降は合同審査区分の影響があり得る（未確認）。神経系（中区分 46）のように新表が旧区分と別の軸で切り直す場合、旧ラベルでは新しい境界の妥当性を検証できない。</p>
"""

DISCUSSION = """
<h2 class="section">4. 考察</h2>
<p><b>中区分の枠が選択を縛っている</b>　統合の相手はほぼ例外なく同じ中区分の中から選ばれている。これは「中区分は維持を基本とする」という見直し方針の帰結だが、内容上の最近傍が別の中区分にある例が複数ある。数学基礎（最多近傍は代数学と情報学基礎論）、国際法（国際関係論）、認知脳科学（神経機能学・実験心理学）、中国文学（日本文学）がそれにあたる。中区分内では「2 番目に近い」相手に収まっており、不合理ではないが、地形から見れば次善の選択である。</p>
<p><b>越境した唯一の例が最も整合的</b>　生物資源保全学（大区分 F 農学）から自然共生システム（大区分 K→新 M 環境）への吸収は、今回唯一の大区分をまたぐ移動であり、28 ペア中で最も区別しにくい組（精度 0.75）だった。制度が枠を越えたとき、データもそれを支持している。</p>
<p><b>不整合 4 件の構図</b>　基礎法学＋刑事法学は、刑事法学が中区分内のどの区分からも独立した島であり、基礎法学から見て最も遠い相手である。理論経済学＋経済学説は、経済学説の内容上の隣が経済史であることを示す。日本語学＋英語学は、二つの「○語学」がそれぞれ言語学と最も混ざっており、内容上の自然な統合先は言語学である。医療技術評価学は内容が機器寄り（医用システムに最も近い）だが社会医学側に送られた。いずれも「小さい区分に家を与える」という運営上の判断として理解できるが、内容の近さは判断材料になっていないように見える。</p>
<p><b>連続体の切り直し</b>　神経系（中区分 46）は旧 3 区分が互いにほとんど分かれず（精度 0.66–0.85）、区分だけで UMAP を計算し直しても完全に混ざる。新表はこれを「分子・細胞レベル」と「システム・個体レベル」という別の軸で 2 分する。旧ラベルでは検証できないが、連続体をどこで切っても境界は恣意的になる領域であることは分かる。令和 10 年度以降の実データで、新しい切り方が応募者の選択と一致するかを追試する価値がある。</p>
<p><b>今後</b>　同じ方法で、新設「量子情報」の課題が地図のどこに座るか（制度は 5 大区分に分属させた）、新旧の大区分分割がどちらが内容的にまとまっているか、内容の例の増減語（+450／−383）が地図上のどの領域にあるかを検証できる。令和 10 年度のデータが入れば、地図から予測した区分選択と実際の選択を比べる追試も可能になる。</p>

<h2>付録 A. 再現</h2>
<p>スクリプト <code>scripts/kubun2028_merges.py</code>（数表と図）、<code>scripts/kubun2028_report.py</code>（本ノート）。入力は <code>data/processed/embeddings.parquet</code>、<code>data/interim/knn50.npz</code>（k=50 近傍。<code>scripts/compute_indicators.py knn</code> で生成）、<code>data/interim/awards.parquet</code>、<code>data/reference/kubun_table.csv</code>。統合の組は <code>data/reference/shinsa_kubun_r10_mext_20260729.txt</code> 参考 2「新旧対照表」（p.142〜）から転記した。</p>
<p class="foot">出典: KAKEN：科学研究費助成事業データベース（国立情報学研究所）のデータを編集・加工。改正資料: 文部科学省 科学技術・学術審議会 学術分科会 科学研究費補助金審査部会「科学研究費助成事業『審査区分表』の改正等について」（令和 8 年 7 月 29 日）https://www.mext.go.jp/a_menu/shinkou/hojyo/1385136_00008.htm 。埋め込み: cl-nagoya/ruri-v3-310m。作成: KAKEN-ATLAS（JSPS 科研費 26K15524）。研究者名・所属機関は用いていない。</p>
"""


def main() -> None:
    body = subprocess.run(["pandoc", str(MD), "-f", "gfm", "-t", "html"], capture_output=True, text=True, check=True).stdout
    # md の見出し: h1（タイトル）→捨てる。「所見」節→ 3. 結果の総括。中区分の節 → 各節に図を差し込む
    body = re.sub(r"<h1[^>]*>.*?</h1>\s*", "", body, count=1, flags=re.S)
    findings = subprocess.run(["pandoc", str(FINDINGS), "-f", "gfm", "-t", "html"], capture_output=True, text=True, check=True).stdout
    body = "<h2>3. 結果</h2><h3>3.1 総括</h3>" + findings + body
    # 統合表・小区分表に class
    body = re.sub(r"<table>(\s*<thead>\s*<tr[^>]*>\s*<th[^>]*>統合</th>)", r'<table class="merge">\1', body)
    body = re.sub(r"<table>(\s*<thead>\s*<tr[^>]*>\s*<th[^>]*>小区分</th>)", r'<table class="codes">\1', body)
    # 判定の色分け
    for word, cls in [("内容と整合", "ok"), ("部分的", "part"), ("不整合", "ng"), ("判定不能", "na")]:
        body = body.replace(f"<p><strong>{word}", f'<p class="{cls}"><strong>{word}')
    # 中区分の節
    n = 0
    def repl(m):
        nonlocal n
        n += 1
        title = m.group(1)
        chu = int(re.search(r"中区分 (\d+)", title).group(1))
        img = FIG / f"kubun2028_chu{chu:02d}.png"
        return (f'<h3 class="section">3.{n + 1} {title}</h3>'
                f'<figure><img src="file://{img}"><figcaption>図 3.{n + 1}　左: 全体地図（2D UMAP）の該当領域。右: この中区分の課題だけで UMAP を再計算。★は統合・吸収に関わる区分。</figcaption></figure>')
    body = re.sub(r"<h2[^>]*>(中区分 [^<]*)</h2>", repl, body)
    # 全ペア表に class
    body = body.replace("<p>中区分内の全ペアの分類精度（低いほど混ざっている）:</p>\n<table>", "<p>中区分内の全ペアの分類精度（低いほど混ざっている）:</p>\n<table class=\"pairs\">")
    # 冒頭の説明段落（md の指標説明）は方法節と重複するので削る
    body = re.sub(r"<p>計算は 768 次元.*?</ul>\s*", "", body, count=1, flags=re.S)
    html = f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>審査区分表 2028 改正の内容的整合</title><style>{CSS}</style></head><body>{FRONT}{body}{DISCUSSION}</body></html>'
    HTML.write_text(html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF.resolve()}", f"file://{HTML.resolve()}"], check=True, capture_output=True)
    print(PDF, PDF.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
