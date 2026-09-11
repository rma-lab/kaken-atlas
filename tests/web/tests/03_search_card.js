// 検索ヒット → カード（スマホ 3D/球面は結果を閉じてカメラが点へ、PC 2D はカードの縁と見出しが点の色）
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  for (const [label, view, device] of [['phone globe', 'globe', 'phone'], ['phone 3D', 'map3d', 'phone'], ['PC 2D', 'map2d', 'pc']]) {
    const p = await lib.open(browser, url(`/${view}/`), device);
    await lib.ready(p); await lib.sleep(800);
    const cam0 = await p.evaluate(() => { const s = plot._fullLayout.scene; return s ? JSON.stringify(s._scene.getCamera().eye) : JSON.stringify(plot._fullLayout.xaxis.range); });
    const hit = await lib.searchAndOpenFirst(p, device, '重力波');
    const st = await p.evaluate(() => {
      const r = document.getElementById('ka-results'), card = document.getElementById('ka-card'), ring = document.getElementById('ka-ring');
      const s = plot._fullLayout.scene; const hd = card.querySelector('div[style*="font-weight:600"]');
      const g = plot._dbgView ? plot._dbgView().sel : null;
      return { results: r.style.display !== 'none', card: card.style.display !== 'none', ring: ring && ring.style.display !== 'none',
               cam: s ? JSON.stringify(s._scene.getCamera().eye) : JSON.stringify(plot._fullLayout.xaxis.range),
               border: card.style.borderColor, header: hd ? hd.style.backgroundColor : null, pointColor: g != null ? colorOf[g] : null };
    });
    const norm = (v) => String(v).replace(/\s/g, '');
    const moved = st.cam !== cam0;
    if (device === 'phone') out.push({ name: `${label} 検索→カード（結果閉じる・カメラ移動）`, ok: hit && st.card && st.ring && !st.results && moved, info: `card=${st.card} results=${st.results} moved=${moved}` });
    else out.push({ name: `${label} 検索→カード（縁・見出し＝点の色）`, ok: hit && st.card && st.pointColor && norm(st.border) === norm(st.pointColor) && norm(st.header) === norm(st.pointColor), info: `border=${st.border} point=${st.pointColor}` });
    await p.close();
  }
  return out;
};
