// 共通部品: docs/ をローカル HTTP で配信し、ヘッドレス Chrome（Metal で GPU）でページを開く。
// 各テストは async (ctx) => [{name, ok, info}] を返す。ctx = {browser, serve, url, open, sleep, ready, out, log}
const puppeteer = require('puppeteer-core');
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const DOCS = process.env.KA_DOCS ? path.resolve(process.env.KA_DOCS) : path.join(ROOT, 'docs');
const OUT = path.join(__dirname, 'out');
const CHROME = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const GL_ARGS = process.env.KA_GL === 'swiftshader'
  ? ['--use-gl=swiftshader', '--enable-unsafe-swiftshader']
  : ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'];  // 深度・描画の検証は GPU が要る
const MIME = { '.html': 'text/html; charset=utf-8', '.json': 'application/json', '.bin': 'application/octet-stream',
               '.js': 'application/javascript', '.png': 'image/png', '.jpg': 'image/jpeg', '.webmanifest': 'application/manifest+json' };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// roots: {name: dir}。URL は http://127.0.0.1:PORT/<name>/... 。root を差し替えられる（更新直後の再現用）
function serve(roots, opts) {
  const state = { roots: Object.assign({}, roots), cache: (opts && opts.cacheControl) || 'max-age=600' };
  const server = http.createServer((req, res) => {
    const u = decodeURIComponent(req.url.split('?')[0]);
    const m = u.match(/^\/([^/]+)(\/.*)$/);
    if (!m || !state.roots[m[1]]) { res.writeHead(404); res.end(); return; }
    let p = m[2]; if (p.endsWith('/')) p += 'index.html';
    fs.readFile(path.join(state.roots[m[1]], p), (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream', 'Cache-Control': state.cache });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => {
    const port = server.address().port;
    resolve({ port, url: (name, rest) => `http://127.0.0.1:${port}/${name}${rest || '/'}`, setRoot: (n, d) => { state.roots[n] = d; }, close: () => server.close() });
  }));
}

async function launch() {
  return puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'].concat(GL_ARGS) });
}

// ページを開く。device: 'pc' | 'phone'（iPhone 13 相当、タッチ）
async function open(browser, url, device, opts) {
  const p = await browser.newPage();
  if (device === 'phone') await p.emulate(puppeteer.KnownDevices['iPhone 13']);
  else await p.setViewport(Object.assign({ width: 1280, height: 820 }, (opts && opts.viewport) || {}));
  p._errors = [];
  p.on('pageerror', (e) => p._errors.push(e.message));
  await p.goto(url, { waitUntil: 'load' });
  return p;
}
// 描画済み（Plotly の layout ができた）まで待つ
const plotted = (p) => p.waitForFunction(() => { const el = document.getElementById('plot'); if (!el || !el._fullLayout) return false; return el._fullLayout.scene ? !!el._fullLayout.scene._scene : !!el._fullLayout.xaxis; }, { timeout: 180000 });
// 詳細データまで読み込み完了（検索欄が有効）
const ready = (p) => p.waitForFunction(() => { const q = document.getElementById('ka-q'); return q && !q.disabled; }, { timeout: 180000 });
const hideChrome = (p) => p.evaluate(() => { ['ka-load', 'ka-phase2'].forEach((id) => { const e = document.getElementById(id); if (e) e.style.display = 'none'; }); });
const aspect = (p) => p.evaluate(() => { const xa = plot._fullLayout.xaxis, ya = plot._fullLayout.yaxis; const ux = (xa.range[1] - xa.range[0]) / xa._length, uy = (ya.range[1] - ya.range[0]) / ya._length; return { ratio: +(uy / ux).toFixed(4), xr: xa.range.map((v) => +v.toFixed(2)), yr: ya.range.map((v) => +v.toFixed(2)) }; });
const cardState = (p) => p.evaluate(() => { const c = document.getElementById('ka-card'); if (!c || c.style.display === 'none') return null; return { text: c.textContent.slice(0, 40), award: (c.textContent.match(/\d\dK\w{5}|\d\dH\w{5}|\d\dJ\w{5}|\d\dKK\w{4}/) || [])[0], url: location.search, border: c.style.borderColor }; });

async function searchAndOpenFirst(p, device, query) {
  await p.click('#ka-q'); await p.type('#ka-q', query); await sleep(900);
  const hit = await p.$('.ka-hit'); if (!hit) return false;
  if (device === 'phone') await hit.tap(); else await hit.click();
  await sleep(1800);
  return true;
}

module.exports = { ROOT, DOCS, OUT, sleep, serve, launch, open, plotted, ready, hideChrome, aspect, cardState, searchAndOpenFirst, puppeteer };
