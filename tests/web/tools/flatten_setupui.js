// リファクタリング用の一回きりの変換: setupUI の巨大な閉包を「script スコープの状態変数＋関数」と
// 「節ごとの init 関数」に平坦化する（動作は変えない）。
//   - setupUI 直下の function 宣言 → script スコープへ（そのまま）
//   - setupUI 直下の var 宣言 → script スコープに名前だけ宣言し、初期化子は init 関数内の代入に
//   - それ以外の文 → 節（// ---- 見出し）ごとの init 関数へ。setupUI は init を順に呼ぶだけに
// これで「setupUI 内の var x = null が main の代入を巻き上げで上書きする」罠（2026-09-10 に 2 回踏んだ）が構造的に消える。
//   node tests/web/tools/flatten_setupui.js scripts/web/index.template.html
const fs = require('fs');
const path = require('path');
const acorn = require(path.join(__dirname, '..', 'node_modules', 'acorn'));

const file = process.argv[2];
const html = fs.readFileSync(file, 'utf8');
const s0 = html.lastIndexOf('<script>') + 8, e0 = html.lastIndexOf('</script>');
const js = html.slice(s0, e0);
const ast = acorn.parse(js.replace('__MANIFEST__', '{}'.padEnd('__MANIFEST__'.length)), { ecmaVersion: 2020, sourceType: 'script' });
const setup = ast.body.find((n) => n.type === 'FunctionDeclaration' && n.id.name === 'setupUI');
if (!setup) throw new Error('setupUI not found');

// 節の名前（// ---- 見出し の出現順）
const SECTION_NAMES = {
  'ヘッダーバー': 'initHeader', '色の見方': 'initColorLegend', '点の詳細表示': 'initPointDetail',
  'クリック/タップ → 選択': 'initSelection', 'タッチ端末: 地図を動かしても選択は閉じない': 'initTouchKeepSelection',
  '2Dタッチ端末のタップ処理': 'initTap2d', '検索': 'initSearch', '種目フィルタ': 'initCategoryFilter',
  '囲って集計': 'initLasso', 'ボトムシート': 'initSheets', '2本指ピンチズーム': 'initPinch2d', '3D/球面のタッチ操作': 'initTouch3d',
};
function sectionName(header) {
  for (const k of Object.keys(SECTION_NAMES)) if (header.includes(k)) return SECTION_NAMES[k];
  throw new Error('unknown section: ' + header);
}

const body = setup.body.body;
const bodyStart = setup.body.start + 1;  // '{' の次
const sections = [];  // {name, header, vars:[], fns:[], init:[]}
let cur = { name: 'initHome', header: '// ---- 読み込み時の範囲／カメラを「全体」として記録 ----', vars: [], fns: [], init: [] };
sections.push(cur);
let prevEnd = bodyStart;
for (const n of body) {
  let gap = js.slice(prevEnd, n.start);
  // 文の行末コメントは文の一部として扱う（次の文の余白に混ざらないように）
  let end = n.end;
  const eol = js.indexOf('\n', end);
  if (eol > 0 && /^\s*\/\/.*$/.test(js.slice(end, eol))) end = eol;
  prevEnd = end;
  // 節の見出し（// ---- …）があれば新しい節へ。余白に見出しが複数あれば順に節を作る（中身のない節は見出しだけ残す）
  const lines = gap.split('\n');
  const heads = lines.map((l, i) => (/^\/\/ ---- /.test(l) ? i : -1)).filter((i) => i >= 0);
  if (heads.length) {
    heads.forEach((hi, k) => {
      const next = k + 1 < heads.length ? heads[k + 1] : lines.length;
      // 見出しに続くコメント行（次の見出しか空行/文まで）を見出しの一部として保持
      let j = hi + 1;
      while (j < next && /^\s*\/\//.test(lines[j])) j++;
      cur = { name: sectionName(lines[hi]), header: lines.slice(hi, j).join('\n'), vars: [], fns: [], init: [] };
      sections.push(cur);
      if (k === heads.length - 1) gap = lines.slice(j).join('\n');
    });
  }
  const gapTrim = gap.replace(/^\n+/, '').replace(/\n+$/, '');
  const lead = gapTrim ? gapTrim + '\n' : '';
  const text = js.slice(n.start, end);
  if (n.type === 'FunctionDeclaration') {
    cur.fns.push(lead + text);
  } else if (n.type === 'VariableDeclaration') {
    const names = n.declarations.map((d) => d.id.name);
    const tail = js.slice(n.end, end).trim();  // 宣言の行末コメント
    const assigns = n.declarations.filter((d) => d.init).map((d) => `${d.id.name} = ${js.slice(d.init.start, d.init.end)};`);
    // 前置きコメントは、初期化子があれば代入側に、なければ宣言側に付ける（重複させない）
    cur.vars.push((assigns.length ? '' : lead) + 'var ' + names.join(', ') + ';' + (tail ? '  ' + tail : ''));
    if (assigns.length) cur.init.push(lead + assigns.join('\n'));
  } else {
    cur.init.push(lead + text);
  }
}
const tailGap = js.slice(prevEnd, setup.body.end - 1);

function indent(t) { return t.split('\n').map((l) => (l.trim() ? '  ' + l : l)).join('\n'); }
let out = '';
const live = sections.filter((sec) => sec.vars.length || sec.fns.length || sec.init.length);
for (const sec of sections) {
  out += sec.header + '\n';
  if (!live.includes(sec)) { out += '\n'; continue; }
  if (sec.vars.length) out += sec.vars.join('\n') + '\n';
  if (sec.fns.length) out += sec.fns.join('\n\n') + '\n';
  out += `function ${sec.name}() {\n` + (sec.init.length ? indent(sec.init.join('\n')) + '\n' : '') + '}\n\n';
}
out += 'function setupUI() {\n' + live.map((s) => `  ${s.name}();`).join('\n') + '\n}' + tailGap.replace(/\s+$/, '') + '\n';

const before = js.slice(0, setup.start), after = js.slice(setup.end);
const newJs = before + out + after;
// 構文チェック
acorn.parse(newJs.replace('__MANIFEST__', '{}'.padEnd('__MANIFEST__'.length)), { ecmaVersion: 2020, sourceType: 'script' });
fs.writeFileSync(file, html.slice(0, s0) + newJs + html.slice(e0));
console.log('sections:', sections.map((s) => `${s.name}(vars ${s.vars.length}, fns ${s.fns.length}, init ${s.init.length})`).join('\n  '));
