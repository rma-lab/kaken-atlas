// 基準スクリーンショットとの画素差分（リファクタリングの安全網 3）。
//   node shots.js record   # 基準を out/baseline/ に撮る（git 管理外。整理を始める前のコードで撮る）
//   node shots.js check    # 現在の docs/ で撮り直し、基準と比べる（差分画素が 0.3% を超えたら NG）
// 構図: 3 ビュー × PC・iPhone 相当の初期表示、2D のフライトゥ後（カード付き）、色の見方パネル、なげなわの結果。
// 半透明の点の WebGL 描画は同じ機械・同じ GPU なら決定的。差分画像は out/diff/ に出す。
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const lib = require('./lib');

const BASE = path.join(lib.OUT, 'baseline'), CUR = path.join(lib.OUT, 'current'), DIFF = path.join(lib.OUT, 'diff');
const THRESH_PIXEL = 0.003;  // 差分画素の許容割合
const TOL = 24;              // チャンネル差の許容（アンチエイリアスの揺れ）

async function shoot(browser, url, dir) {
  fs.mkdirSync(dir, { recursive: true });
  const shots = [];
  const settle = { map2d: 2500, map3d: 4000, globe: 4000 };
  for (const view of ['map2d', 'map3d', 'globe']) {
    for (const device of ['pc', 'phone']) {
      const p = await lib.open(browser, url(`/${view}/`), device, { viewport: { width: 1200, height: 900 } });
      await lib.ready(p); await lib.sleep(settle[view]); await lib.hideChrome(p);
      await p.evaluate(() => { const b = document.getElementById('ka-bar'); if (b) b.style.visibility = 'hidden'; });
      const f = `${view}_${device}.png`; await p.screenshot({ path: path.join(dir, f) }); shots.push(f);
      if (view === 'map2d' && device === 'pc') {
        await p.evaluate(() => { const b = document.getElementById('ka-bar'); if (b) b.style.visibility = ''; });
        await p.hover('#ka-color-btn'); await lib.sleep(400);
        await p.screenshot({ path: path.join(dir, 'map2d_pc_colorpanel.png') }); shots.push('map2d_pc_colorpanel.png');
        await p.mouse.move(10, 500); await lib.sleep(500);
        await lib.searchAndOpenFirst(p, 'pc', '重力波'); await lib.sleep(600);
        await p.evaluate(() => { const r = document.getElementById('ka-results'); if (r) r.style.display = 'none'; });
        await p.screenshot({ path: path.join(dir, 'map2d_pc_flyto_card.png') }); shots.push('map2d_pc_flyto_card.png');
        await p.keyboard.press('Escape'); await lib.sleep(300);
        await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': [-8.64, 15.23], 'yaxis.range': [-1.1, 15.25] })); await lib.sleep(1500);
        const btn = await p.$('.modebar-btn[data-title="囲って集計（なげなわ）"]'); await btn.click(); await lib.sleep(200);
        const pts = [[520, 380], [640, 360], [700, 450], [640, 540], [520, 520], [520, 380]];
        await p.mouse.move(pts[0][0], pts[0][1]); await p.mouse.down(); for (const [x, y] of pts.slice(1)) await p.mouse.move(x, y, { steps: 3 }); await p.mouse.up(); await lib.sleep(1000);
        await p.screenshot({ path: path.join(dir, 'map2d_pc_lasso.png') }); shots.push('map2d_pc_lasso.png');
      }
      await p.close();
    }
  }
  return shots;
}

function compare(f) {
  const a = PNG.sync.read(fs.readFileSync(path.join(BASE, f))), b = PNG.sync.read(fs.readFileSync(path.join(CUR, f)));
  if (a.width !== b.width || a.height !== b.height) return { ok: false, info: `size ${a.width}x${a.height} vs ${b.width}x${b.height}` };
  const d = new PNG({ width: a.width, height: a.height });
  let diff = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const dr = Math.abs(a.data[i] - b.data[i]), dg = Math.abs(a.data[i + 1] - b.data[i + 1]), db = Math.abs(a.data[i + 2] - b.data[i + 2]);
    const bad = dr > TOL || dg > TOL || db > TOL;
    if (bad) diff++;
    const g = Math.round(0.3 * b.data[i] + 0.59 * b.data[i + 1] + 0.11 * b.data[i + 2]);
    d.data[i] = bad ? 220 : Math.round(200 + g * 0.2); d.data[i + 1] = bad ? 40 : Math.round(200 + g * 0.2); d.data[i + 2] = bad ? 40 : Math.round(200 + g * 0.2); d.data[i + 3] = 255;
  }
  const frac = diff / (a.width * a.height);
  if (frac > 0) { fs.mkdirSync(DIFF, { recursive: true }); fs.writeFileSync(path.join(DIFF, f), PNG.sync.write(d)); }
  return { ok: frac <= THRESH_PIXEL, info: `${(frac * 100).toFixed(3)}% の画素が違う` };
}

(async () => {
  const mode = process.argv[2] || 'check';
  const browser = await lib.launch();
  const srv = await lib.serve({ docs: lib.DOCS });
  const url = (rest) => srv.url('docs', rest);
  if (mode === 'record') {
    fs.rmSync(BASE, { recursive: true, force: true });
    const shots = await shoot(browser, url, BASE);
    console.log(`基準を記録: ${BASE}\n  ` + shots.join('\n  '));
  } else {
    if (!fs.existsSync(BASE)) { console.error('基準がありません。先に  node shots.js record'); process.exit(2); }
    fs.rmSync(CUR, { recursive: true, force: true }); fs.rmSync(DIFF, { recursive: true, force: true });
    const shots = await shoot(browser, url, CUR);
    let ng = 0;
    for (const f of shots) { const r = compare(f); if (!r.ok) ng++; console.log(`${r.ok ? 'OK' : 'NG'}  ${f}  ${r.info}`); }
    console.log(`\n${shots.length - ng}/${shots.length} 一致` + (ng ? `  （差分画像: ${DIFF}）` : ''));
    await browser.close(); srv.close(); process.exit(ng ? 1 : 0);
  }
  await browser.close(); srv.close();
})().catch((e) => { console.error(e); process.exit(2); });
