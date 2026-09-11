// 自前のなげなわ: 囲う→集計が速い、直後に移動できる、囲いが追随、膜の上で操作が通る、閉じる／Esc
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  for (const device of ['phone', 'pc']) {
    const p = await lib.open(browser, url('/map2d/'), device, { viewport: { deviceScaleFactor: 2 } });
    await lib.ready(p); await lib.sleep(1200);
    const btn = await p.$('.modebar-btn[data-title="囲って集計（なげなわ）"]');
    if (!btn) { out.push({ name: `${device} なげなわボタン`, ok: false }); await p.close(); continue; }
    if (device === 'phone') await btn.tap(); else await btn.click(); await lib.sleep(200);
    const pts = device === 'phone' ? [[120, 350], [260, 330], [330, 450], [250, 560], [120, 520], [120, 350]] : [[520, 380], [640, 360], [700, 450], [640, 540], [520, 520], [520, 380]];
    const t0 = Date.now();
    if (device === 'phone') { const ts = p.touchscreen; await ts.touchStart(pts[0][0], pts[0][1]); for (const [x, y] of pts.slice(1)) { await ts.touchMove(x, y); await lib.sleep(20); } await ts.touchEnd(); }
    else { await p.mouse.move(pts[0][0], pts[0][1]); await p.mouse.down(); for (const [x, y] of pts.slice(1)) await p.mouse.move(x, y, { steps: 3 }); await p.mouse.up(); }
    const got = await p.waitForFunction(() => !!document.getElementById('ka-selclear'), { timeout: 20000 }).then(() => true).catch(() => false);
    const dt = Date.now() - t0;
    const s1 = await p.evaluate(() => plot._dbgLasso());
    out.push({ name: `${device} 囲う→集計`, ok: got && !s1.mode && s1.outline && s1.veil && dt < 3000, info: `${dt}ms panel=${s1.panel}` });
    await p.screenshot({ path: `${lib.OUT}/lasso_${device}.png` });
    const xr0 = await p.evaluate(() => plot._fullLayout.xaxis.range.slice());
    if (device === 'phone') { const ts = p.touchscreen; await ts.touchStart(200, 250); await ts.touchMove(250, 290); await lib.sleep(40); await ts.touchMove(300, 330); await ts.touchEnd(); }
    else { await p.mouse.move(300, 300); await p.mouse.down(); await p.mouse.move(360, 330, { steps: 6 }); await p.mouse.up(); }
    await lib.sleep(500);
    const s2 = await p.evaluate(() => ({ xr: plot._fullLayout.xaxis.range, d: plot._dbgLasso() }));
    out.push({ name: `${device} 直後に移動でき、囲いが追随しパネルは残る`, ok: Math.abs(s2.xr[0] - xr0[0]) > 0.01 && s2.d.outline && s2.d.panel === 'block' });
    if (device === 'pc') {
      const under = await p.evaluate(() => { const e = document.elementFromPoint(900, 700); return e.tagName + '.' + (e.getAttribute('class') || ''); });
      await lib.searchAndOpenFirst(p, 'pc', '重力波');
      const card = await lib.cardState(p);
      out.push({ name: 'pc 膜の上でも操作が地図に届く（検索→カード）', ok: /nsewdrag/.test(under) && !!card, info: under });
      await p.keyboard.press('Escape'); await lib.sleep(300);
      const s3 = await p.evaluate(() => plot._dbgLasso());
      out.push({ name: 'pc Esc で囲いとパネルが消える', ok: !s3.outline && s3.panel === 'none' });
    } else {
      const c = await p.$('#ka-selclear'); await c.tap(); await lib.sleep(300);
      const s3 = await p.evaluate(() => plot._dbgLasso());
      out.push({ name: 'phone 閉じる × で囲いとパネルが消える', ok: !s3.outline && s3.panel === 'none' });
    }
    out.push({ name: `${device} ページエラーなし`, ok: p._errors.length === 0, info: p._errors[0] || '' });
    await p.close();
  }
  return out;
};
