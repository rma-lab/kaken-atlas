// ?award=課題番号 で開くと 3 ビューともカードとリングが出る。存在しない番号では何も起きない
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const AWARD = '24K23444';
  for (const [label, view, device] of [['PC 2D', 'map2d', 'pc'], ['PC 3D', 'map3d', 'pc'], ['phone globe', 'globe', 'phone']]) {
    const p = await lib.open(browser, url(`/${view}/?award=${AWARD}`), device);
    await lib.plotted(p); await lib.sleep(view === 'map2d' ? 2500 : 4000);
    const st = await lib.cardState(p);
    const ring = await p.evaluate(() => { const r = document.getElementById('ka-ring'); return r && r.style.display !== 'none'; });
    out.push({ name: `${label} 深リンク`, ok: !!st && st.award === AWARD && ring && st.url === `?award=${AWARD}`, info: st ? `card=${st.award} ring=${ring} url=${st.url}` : 'no card' });
    await p.close();
  }
  const q = await lib.open(browser, url('/map2d/?award=99Z99999'), 'pc');
  await lib.plotted(q); await lib.sleep(1500);
  out.push({ name: '存在しない番号ではカードなし', ok: (await lib.cardState(q)) === null });
  await q.close();
  return out;
};
