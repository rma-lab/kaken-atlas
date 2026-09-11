// Service Worker: 2 回目は端末保存から読み、オフラインでも開いて検索できる
module.exports = async ({ browser, lib }) => {
  const out = [];
  const srv = await lib.serve({ docs: lib.DOCS }, { cacheControl: 'no-store' });  // SW の効果だけを見る
  const ctx = await browser.createBrowserContext();
  const p = await ctx.newPage(); await p.setViewport({ width: 1000, height: 800 });
  await p.goto(srv.url('docs', '/map2d/'), { waitUntil: 'load' }); await lib.ready(p); await lib.sleep(1500);
  await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(1000);
  const caches = await p.evaluate(async () => { const names = await window.caches.keys(); const o = {}; for (const n of names) o[n] = (await (await window.caches.open(n)).keys()).length; return o; });
  const dataCache = Object.keys(caches).find((k) => k.startsWith('ka-data-'));
  out.push({ name: '2 回目でデータがキャッシュ済み', ok: !!dataCache && caches[dataCache] >= 102, info: JSON.stringify(caches) });
  await p.setOfflineMode(true);
  await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(800);
  const ok = await p.evaluate(() => ({ n: M.n, rows: !!(getRow(0) && getRow(M.n - 1)) }));
  await p.click('#ka-q'); await p.type('#ka-q', '重力波'); await lib.sleep(900);
  const hits = await p.evaluate(() => document.querySelectorAll('.ka-hit').length);
  out.push({ name: 'オフラインで開いて検索できる', ok: ok.rows && hits > 0, info: `hits=${hits}` });
  await p.setOfflineMode(false);
  await ctx.close(); srv.close();
  return out;
};
