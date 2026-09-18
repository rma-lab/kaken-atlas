// 地名（既定オフ）: 2D＝初回は出ず案内が出る、ツールバーの「地名」でオン（全体表示でも粗い階層）、拡大で細かい階層、重なりなし・描画領域内、
// チェックでオフ、設定が端末に残る、案内は 2 回目に出ない。球面＝ボタンでオン、回すと入れ替わる、近づくと細かい階層。3D にはない
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
  const hint = (p) => p.evaluate(() => !!document.getElementById('ka-pn-hint'));
  const pressButton = async (p, device) => {
    const b = await p.$('.modebar-btn[data-title="地名を表示（切替）"]');
    if (!b) return false;
    if (device === 'phone') await b.tap(); else await b.click();
    await lib.sleep(400);
    return true;
  };
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
    await lib.ready(p);
    // 他の試験が同じブラウザで先に開いていると「案内は 1 回だけ」の記録が残るので、消してから開き直す
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); localStorage.removeItem('ka-pn-hint'); } catch (_) {} });
    await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(1200);
    const has = await p.evaluate(() => !!M.placenames);
    if (!has) { out.push({ name: `${device} manifest に地名なし（PLACENAMES=0 で生成）`, ok: true, info: 'skip' }); await p.close(); continue; }
    const s0 = await state(p), h0 = await hint(p);
    out.push({ name: `${device} 初回は地名オフで案内が出る`, ok: !s0.on && s0.shown === 0 && h0, info: JSON.stringify(s0) + ' hint=' + h0 });
    const pressed = await pressButton(p, device);
    const s1 = await state(p), b1 = await boxes(p), h1 = await hint(p);
    const lit1 = await p.evaluate(() => document.querySelector('.modebar-btn[data-title="地名を表示（切替）"]').classList.contains('active'));
    out.push({ name: `${device} ボタンでオン（点灯）、全体表示でも粗い階層、案内は消える`, ok: pressed && lit1 && s1.on && s1.level === 0 && s1.shown >= 5 && b1.n === s1.shown && b1.overlap === 0 && b1.outside === 0 && !h1, info: `shown=${s1.shown} lit=${lit1} overlap=${b1.overlap} outside=${b1.outside}` });
    await zoomTo(p, 4);
    const s2 = await state(p), b2 = await boxes(p);
    out.push({ name: `${device} 4 倍で細かい階層`, ok: s2.level === 1 && s2.shown >= 5 && b2.overlap === 0 && b2.outside === 0, info: `shown=${s2.shown} overlap=${b2.overlap} outside=${b2.outside}` });
    await p.screenshot({ path: `${lib.OUT}/placenames_${device}.png` });
    await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': plot._fullLayout.xaxis.range.map((v) => v + 0.7) })); await lib.sleep(400);
    const s3 = await state(p);
    out.push({ name: `${device} 移動後も描かれる`, ok: s3.level === 1 && s3.shown >= 5, info: `shown=${s3.shown}` });
    // 設定が残る: 再読み込み後もオン、案内は出ない
    await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(800);
    const s4 = await state(p), h4 = await hint(p);
    const lit4 = await p.evaluate(() => document.querySelector('.modebar-btn[data-title="地名を表示（切替）"]').classList.contains('active'));
    out.push({ name: `${device} 再読み込み後もオン（点灯）が残り、案内は 2 回目に出ない`, ok: s4.on && lit4 && s4.shown >= 5 && !h4, info: `shown=${s4.shown} lit=${lit4} hint=${h4}` });
    // チェックでオフ（ボタンの点灯も消える）
    await p.evaluate(() => { const c = document.getElementById('ka-pn-toggle'); c.checked = false; c.dispatchEvent(new Event('change')); });
    await lib.sleep(200);
    const s5 = await p.evaluate(() => ({ d: plot._dbgPlacenames(), lit: document.querySelector('.modebar-btn[data-title="地名を表示（切替）"]').classList.contains('active') }));
    out.push({ name: `${device} チェックでオフ、ボタンの点灯も消える`, ok: !s5.d.on && s5.d.shown === 0 && !s5.lit });
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); localStorage.removeItem('ka-pn-hint'); } catch (_) {} });
    out.push({ name: `${device} ページエラーなし`, ok: p._errors.length === 0, info: p._errors[0] || '' });
    await p.close();
  }
  // 球面: 既定オフ、ボタンでオン（粗い階層）、回すと地名が入れ替わる、近づくと細かい階層、重なりなし
  for (const device of ['pc', 'phone']) {
    const p = await lib.open(browser, url('/globe/'), device);
    await lib.ready(p);
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); localStorage.removeItem('ka-pn-hint'); } catch (_) {} });
    await p.reload({ waitUntil: 'load' }); await lib.ready(p); await lib.sleep(2500);
    const has = await p.evaluate(() => !!M.placenames);
    if (!has) { out.push({ name: `${device} 球面 manifest に地名なし`, ok: true, info: 'skip' }); await p.close(); continue; }
    const texts = () => p.evaluate(() => {
      const bs = Array.from(document.querySelectorAll('#ka-placenames text')).map((t) => { const r = t.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; });
      let overlap = 0;
      for (let i = 0; i < bs.length; i++) for (let j = i + 1; j < bs.length; j++) { const a = bs[i], b = bs[j]; if (a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]) overlap++; }
      return { n: bs.length, overlap, words: Array.from(document.querySelectorAll('#ka-placenames text')).map((t) => t.textContent).join('|') };
    });
    const g00 = await state(p), hh = await hint(p);
    out.push({ name: `${device} 球面 初回は地名オフで案内が出る`, ok: !g00.on && g00.shown === 0 && hh });
    const pressed = await pressButton(p, device); await lib.sleep(600);
    const g0 = await state(p), t0 = await texts();
    out.push({ name: `${device} 球面 ボタンでオン、粗い階層`, ok: pressed && g0.on && g0.level === 0 && g0.shown >= 3 && t0.n === g0.shown && t0.overlap === 0, info: `shown=${g0.shown} overlap=${t0.overlap}` });
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
    out.push({ name: `${device} 球面 チェックでオフ`, ok: !g3.on && g3.shown === 0 });
    await p.evaluate(() => { try { localStorage.removeItem('ka-placenames'); localStorage.removeItem('ka-pn-hint'); } catch (_) {} });
    out.push({ name: `${device} 球面 ページエラーなし`, ok: p._errors.length === 0, info: p._errors[0] || '' });
    await p.close();
  }
  const p3 = await lib.open(browser, url('/map3d/'), 'pc');
  await lib.plotted(p3); await lib.sleep(800);
  const s3d = await p3.evaluate(() => ({ d: plot._dbgPlacenames(), btn: !!document.querySelector('.modebar-btn[data-title="地名を表示（切替）"]') }));
  out.push({ name: '3D には地名層もボタンもない', ok: !s3d.d.svg && s3d.d.shown === 0 && !s3d.btn });
  await p3.close();
  return out;
};
