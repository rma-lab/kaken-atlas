// 更新直後の再現: 旧版（タグ v1.2）の docs を配信して Service Worker に旧データを保存させ、配信元を現行に切り替えて
// 1 回目の再読み込みで正常に描画できるか（データ URL の ?v= 版の効果）
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
module.exports = async ({ browser, lib }) => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ka-old-'));
  try { execSync(`git -C "${lib.ROOT}" archive v1.2 docs | tar -x -C "${tmp}"`, { stdio: 'ignore' }); }
  catch (e) { return [{ name: '旧版 docs の取り出し（git archive v1.2）', ok: false, info: String(e).split('\n')[0] }]; }
  const srv = await lib.serve({ site: path.join(tmp, 'docs') });
  const ctx = await browser.createBrowserContext();
  const p = await ctx.newPage(); await p.setViewport({ width: 1000, height: 800 });
  await p.goto(srv.url('site', '/map2d/'), { waitUntil: 'load' }); await lib.ready(p); await lib.sleep(1500);
  await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(500);
  const oldOk = await p.evaluate(() => !!navigator.serviceWorker.controller && !M.textColor);
  srv.setRoot('site', lib.DOCS);
  await p.reload({ waitUntil: 'load' });
  const st = await Promise.race([lib.ready(p).then(() => 'ready'), p.waitForFunction(() => /失敗/.test(document.getElementById('ka-load-msg').textContent), { timeout: 180000 }).then(() => 'error')]);
  const info = await p.evaluate(() => ({ textColor: !!M.textColor, colors: typeof colorOf !== 'undefined' && colorOf ? colorOf.length : null, n: M.n, msg: document.getElementById('ka-load-msg').textContent.slice(0, 50) }));
  await ctx.close(); srv.close(); fs.rmSync(tmp, { recursive: true, force: true });
  return [{ name: '旧版を保存後、新版へ切替の 1 回目で正常', ok: oldOk && st === 'ready' && info.colors === info.n, info: `${st} ${info.msg}` }];
};
