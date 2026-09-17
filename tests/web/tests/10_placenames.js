// キーワード地名: 2D＝全体表示では出ない、拡大で粗い階層→細かい階層、重なりなし・描画領域内、トグルで消える（端末に記憶）。
// 球面＝既定で粗い階層、回すと入れ替わる、近づくと細かい階層。3D にはない
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const zoomTo = async (p, z) => {  // 全体表示の幅 ÷ z の範囲を地図の中央付近に（等スケール）
    await p.evaluate((z) => {
      const w = (homeRange.x[1] - homeRange.x[0]) / z, cx = 3.5, cy = 7.5;
      const hy = w / 2 * plot._fullLayout.yaxis._length / plot._fullLayout.xaxis._length;
      return Plotly.relayout(plot, { 'xaxis.range': [cx - w / 2, cx + w / 2], 'yaxis.range': [cy - hy, cy + hy] });
    }, z);
    await lib.sleep(500);
  };
  const state = (p) => p.evaluate(() => plot._dbgPlacenames());
  // 描かれた地名の箱（tspan の実寸）: 互いに重ならず、描画領域の内側にあること
  const boxes = (p) => p.evaluate(() => {
    const sz = plot._fullLayout._size, area = [sz.l, sz.t, sz.l + sz.w, sz.t + sz.h];
    const r0 = plot.getBoundingClientRect();
    const bs = Array.from(document.querySelectorAll('#ka-placenames text')).map((t) => { const r = t.getBoundingClientRect(); return [r.left - r0.left, r.top - r0.top, r.right - r0.left, r.bottom - r0.top]; });
    let overlap = 0, outside = 0;
    for (let i = 0; i < bs.length; i++) {
      const a = bs[i];
      if (a[0] < area[0] - 1 || a[1] < area[1] - 1 || a[2] > area[2] + 1 || a[3] > area[3] + 1) outside++;
      for (let j = i + 1; j < bs.length; j++) { const b = bs[j]; if (a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]) overlap++; }
    }
    return { n: bs.length, overlap, outside };
  });
  for (const device of ['pc', 'phone']) {
    const p = await lib.open(browser, url('/map2d/'), device);
    await lib.ready(p); await lib.sleep(1200);
    const has = await p.evaluate(() => !!M.placenames);
    if (!has) { out.push({ name: `${device} manifest に地名なし（PLACENAMES=0 で生成）`, ok: true, info: 'skip' }); await p.close(); continue; }
    const s0 = await state(p);
    out.push({ name: `${device} 全体表示では地名なし`, ok: s0.on && s0.level === -1 && s0.shown === 0, info: JSON.stringify(s0) });
    await zoomTo(p, 2);
    const s1 = await state(p), b1 = await boxes(p);
    out.push({ name: `${device} 2 倍で粗い階層`, ok: s1.level === 0 && s1.shown >= 5 && b1.n === s1.shown && b1.overlap === 0 && b1.outside === 0, info: `shown=${s1.shown} overlap=${b1.overlap} outside=${b1.outside}` });
    await zoomTo(p, 4);
    const s2 = await state(p), b2 = await boxes(p);
    out.push({ name: `${device} 4 倍で細かい階層`, ok: s2.level === 1 && s2.shown >= 5 && b2.overlap === 0 && b2.outside === 0, info: `shown=${s2.shown} overlap=${b2.overlap} outside=${b2.outside}` });
    await p.screenshot({ path: `${lib.OUT}/placenames_${device}.png` });
    // ドラッグ（移動）中も追随して描かれる: 範囲を少しずらす
    await zoomTo(p, 4); await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': plot._fullLayout.xaxis.range.map((v) => v + 0.7) })); await lib.sleep(400);
    const s3 = await state(p);
    out.push({ name: `${device} 移動後も描かれる`, ok: s3.level === 1 && s3.shown >= 5, info: `shown=${s3.shown}` });
    // トグル OFF → 消える、再読み込み後も OFF が残る
    await p.evaluate(() => { const c = document.getElementById('ka-pn-toggle'); c.checked = false; c.dispatchEvent(new Event('change')); });
    await lib.sleep(200);
    const s4 = await state(p);
    out.push({ name: `${device} トグル OFF で消える`, ok: !s4.on && s4.shown === 0 });
    await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(800); await zoomTo(p, 2);
    const s5 = await state(p);
    out.push({ name: `${device} 再読み込み後も OFF が残る`, ok: !s5.on && s5.shown === 0 });
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); } catch (_) {} });
    out.push({ name: `${device} ページエラーなし`, ok: p._errors.length === 0, info: p._errors[0] || '' });
    await p.close();
  }
  // 球面: 既定の表示で粗い階層、回すと地名が入れ替わる、近づくと細かい階層、重なりなし、トグルで消える
  for (const device of ['pc', 'phone']) {
    const p = await lib.open(browser, url('/globe/'), device);
    await lib.ready(p); await lib.sleep(2500);
    const has = await p.evaluate(() => !!M.placenames);
    if (!has) { out.push({ name: `${device} 球面 manifest に地名なし`, ok: true, info: 'skip' }); await p.close(); continue; }
    const texts = () => p.evaluate(() => {
      const bs = Array.from(document.querySelectorAll('#ka-placenames text')).map((t) => { const r = t.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; });
      let overlap = 0;
      for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) { const a = bs[i], b = bs[j]; if (a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]) overlap++; }
      return { n: bs.length, overlap, words: Array.from(document.querySelectorAll('#ka-placenames text')).map((t) => t.textContent).join('|') };
    });
    const g0 = await state(p), t0 = await texts();
    out.push({ name: `${device} 球面 既定の表示で粗い階層`, ok: g0.on && g0.level === 0 && g0.shown >= 3 && t0.n === g0.shown && t0.overlap === 0, info: `shown=${g0.shown} overlap=${t0.overlap}` });
    await p.evaluate(() => { const c = liveCamera(); const d = Math.hypot(c.eye.x, c.eye.y, c.eye.z); Plotly.relayout(plot, { 'scene.camera.eye': { x: 0, y: 0.05 * d, z: -d }, 'scene.camera.up': { x: 0, y: 1, z: 0 } }); });
    await lib.sleep(1800);
    const g1 = await state(p), t1 = await texts();
    out.push({ name: `${device} 球面 回すと地名が入れ替わる`, ok: g1.shown >= 3 && t1.overlap === 0 && t1.words !== t0.words, info: `shown=${g1.shown}` });
    await p.evaluate(() => { const c = liveCamera(); Plotly.relayout(plot, { 'scene.camera.eye': { x: c.eye.x * 0.62, y: c.eye.y * 0.62, z: c.eye.z * 0.62 } }); });
    await lib.sleep(1800);
    const g2 = await state(p), t2 = await texts();
    out.push({ name: `${device} 球面 近づくと細かい階層`, ok: g2.level === 1 && g2.shown >= 5 && t2.overlap === 0, info: `shown=${g2.shown} overlap=${t2.overlap}` });
    await p.screenshot({ path: `${lib.OUT}/placenames_globe_${device}.png` });
    await p.evaluate(() => { const c = document.getElementById('ka-pn-toggle'); c.checked = false; c.dispatchEvent(new Event('change')); });
    await lib.sleep(300);
    const g3 = await state(p);
    out.push({ name: `${device} 球面 トグル OFF で消える`, ok: !g3.on && g3.shown === 0 });
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); } catch (_) {} });
    out.push({ name: `${device} 球面 ページエラーなし`, ok: p._errors.length === 0, info: p._errors[0] || '' });
    await p.close();
  }
  const p3 = await lib.open(browser, url('/map3d/'), 'pc');
  await lib.plotted(p3); await lib.sleep(800);
  const s3d = await p3.evaluate(() => plot._dbgPlacenames());
  out.push({ name: '3D には地名層がない', ok: !s3d.svg && s3d.shown === 0 });
  await p3.close();
  return out;
};
