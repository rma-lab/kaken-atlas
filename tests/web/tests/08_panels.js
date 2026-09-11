// 「色の見方」パネル（PC はヘッダー、スマホは羅針盤メニュー）と、スマホの大区分シート
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const p = await lib.open(browser, url('/map2d/'), 'pc', { viewport: { deviceScaleFactor: 2 } });
  await lib.plotted(p); await lib.sleep(1500); await lib.hideChrome(p);
  await p.hover('#ka-color-btn'); await lib.sleep(400);
  const pc = await p.evaluate(() => { const b = document.getElementById('ka-color-body'); return { shown: b.style.display === 'block', paths: b.querySelectorAll('path').length, words: b.querySelectorAll('text tspan').length }; });
  out.push({ name: 'PC 色の見方（色相環 12 方位＋特徴語）', ok: pc.shown && pc.paths === 12 && pc.words >= 20, info: JSON.stringify(pc) });
  await p.screenshot({ path: `${lib.OUT}/panels_pc_color.png` });
  await p.close();
  const q = await lib.open(browser, url('/map2d/'), 'phone', { viewport: { deviceScaleFactor: 2 } });
  await lib.plotted(q); await lib.sleep(1500); await lib.hideChrome(q);
  await q.tap('#ka-home-btn'); await lib.sleep(300);
  const items = await q.evaluate(() => Array.from(document.getElementById('ka-home-body').children).map((e) => e.textContent.trim()).filter(Boolean));
  out.push({ name: 'phone 羅針盤メニューに 色の見方・操作の説明', ok: items.includes('色の見方') && items.includes('操作の説明'), info: items.join('|') });
  const link = await q.evaluateHandle(() => Array.from(document.getElementById('ka-home-body').querySelectorAll('div')).find((e) => e.textContent.trim() === '色の見方'));
  await link.tap(); await lib.sleep(400);
  out.push({ name: 'phone 色の見方が開く', ok: (await q.evaluate(() => document.getElementById('ka-color-body').style.display)) === 'block' });
  await q.screenshot({ path: `${lib.OUT}/panels_phone_color.png` });
  await q.tap('#plot'); await lib.sleep(300);
  await q.tap('#ka-dai-btn'); await lib.sleep(500);
  const sheet = await q.evaluate(() => { const s = document.getElementById('ka-dai-drawer'); return { open: s && s.style.transform === 'translateY(0)' || (s && s.style.transform === 'translateY(0px)'), rows: s ? s.querySelectorAll('.ka-sheet-item').length : 0 }; });
  out.push({ name: 'phone 大区分シート（13 行）', ok: !!sheet.open && sheet.rows === 13, info: JSON.stringify(sheet) });
  await q.screenshot({ path: `${lib.OUT}/panels_phone_dai.png` });
  await q.close();
  return out;
};
