// 地図サイトの回帰試験を全部回す:  cd tests/web && npm install && node run.js [名前の一部...]
// 各テストは tests/*.js。結果行は OK / NG。1 つでも NG なら終了コード 1。
const fs = require('fs');
const path = require('path');
const lib = require('./lib');

(async () => {
  const filter = process.argv.slice(2);
  const files = fs.readdirSync(path.join(__dirname, 'tests')).filter((f) => f.endsWith('.js')).sort()
    .filter((f) => !filter.length || filter.some((k) => f.includes(k)));
  fs.mkdirSync(lib.OUT, { recursive: true });
  const t0 = Date.now();
  const browser = await lib.launch();
  const srv = await lib.serve({ docs: lib.DOCS });
  const ctx = { browser, srv, url: (rest) => srv.url('docs', rest), lib };
  let ng = 0, total = 0;
  for (const f of files) {
    const name = f.replace(/\.js$/, '');
    const t1 = Date.now();
    let results;
    try { results = await require(path.join(__dirname, 'tests', f))(ctx); }
    catch (e) { results = [{ name: 'crash', ok: false, info: String(e && e.stack || e).split('\n')[0] }]; }
    for (const r of results) {
      total++; if (!r.ok) ng++;
      console.log(`${r.ok ? 'OK' : 'NG'}  ${name} / ${r.name}${r.info ? '  ' + r.info : ''}`);
    }
    console.log(`    (${name}: ${((Date.now() - t1) / 1000).toFixed(0)}s)`);
  }
  await browser.close(); srv.close();
  console.log(`\n${total - ng}/${total} OK  (${((Date.now() - t0) / 60000).toFixed(1)} 分)`);
  process.exit(ng ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
