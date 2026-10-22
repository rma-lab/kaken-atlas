// 球面のカメラ操作（Google マップ流、2026-10-22）: Plotly のマウス操作が切れている、ドラッグは画面中央の面の点がポインタに 1:1 で付いてくる、
// 寄ると同じドラッグで回る角が小さい、ホイールはポインタの下の面の点を留めたまま寄る／離れる、ダブルクリックで全体に戻る、
// スマホのピンチは指の間の面の点を留めたまま寄る
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const dist = (p) => p.evaluate(() => { const c = liveCamera(); return Math.hypot(c.eye.x, c.eye.y, c.eye.z); });
  const eyeDir = (p) => p.evaluate(() => { const c = liveCamera(); const d = Math.hypot(c.eye.x, c.eye.y, c.eye.z); return [c.eye.x / d, c.eye.y / d, c.eye.z / d]; });
  const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
  const center = (p) => p.evaluate(() => { const r = plot.querySelector('canvas').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
  const surf = (p, x, y) => p.evaluate((x, y) => plot._surfacePoint(x, y), x, y);
  const screenOf = (p, S) => p.evaluate((S) => plot._camToScreen(S), S);
  const off = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  // 1:1 のドラッグ（最後の移動から 150ms 置いて離す＝慣性なし）
  const drag = async (p, x0, y0, dx, dy) => {
    await p.mouse.move(x0, y0); await p.mouse.down();
    for (let i = 1; i <= 10; i++) { await p.mouse.move(x0 + dx * i / 10, y0 + dy * i / 10); await lib.sleep(16); }
    await lib.sleep(150); await p.mouse.up(); await lib.sleep(500);
  };

  // ---- PC ----
  {
    const p = await lib.open(browser, url('/globe/'), 'pc');
    await lib.plotted(p); await lib.sleep(1500);
    const kb = await p.evaluate(() => plot._fullLayout.scene._scene.camera.keyBindingMode);
    out.push({ name: 'pc 球面 Plotly のマウス操作が切れている（dragmode=false）', ok: kb === false, info: `keyBindingMode=${kb}` });
    const [cx, cy] = await center(p);
    // 画素→視線→球面→画素 の往復（fovy・縦横比の仮定の検算）
    const S0 = await surf(p, cx + 100, cy + 60);  // 全体表示の球は半径 170px ほど（PC）。その内側の画素で
    const back = S0 && await screenOf(p, S0);
    out.push({ name: 'pc 球面 画素→面の点→画素の往復が 1px 以内', ok: !!S0 && off(back, [cx + 100, cy + 60]) < 1, info: S0 ? `誤差=${off(back, [cx + 100, cy + 60]).toFixed(2)}px` : '面に当たらない' });
    // ドラッグ: 中央の面の点が (40, 16) 動く（1:1 は一次近似。全体表示では 40px で 11° 回り、ずれは 1px 未満）
    const d0 = await dist(p), e0 = await eyeDir(p);
    const Sc = await surf(p, cx, cy);
    await drag(p, cx, cy, 40, 16);
    const sc1 = await screenOf(p, Sc), e1 = await eyeDir(p);
    const a1 = angle(e0, e1);
    out.push({ name: 'pc 球面 ドラッグで中央の面の点がポインタに付いてくる（3px 以内）', ok: off(sc1, [cx + 40, cy + 16]) < 3, info: `ずれ=${off(sc1, [cx + 40, cy + 16]).toFixed(1)}px 回転=${(a1 * 180 / Math.PI).toFixed(1)}°` });
    out.push({ name: 'pc 球面 ドラッグ後も視距離は不変', ok: Math.abs(await dist(p) - d0) < 1e-6 });
    // ホイール: ポインタの下の面の点が留まったまま寄る
    const px = cx + 90, py = cy + 70;
    const S1 = await surf(p, px, py);
    await p.mouse.move(px, py);
    for (let i = 0; i < 4; i++) { await p.mouse.wheel({ deltaY: -100 }); await lib.sleep(60); }
    await lib.sleep(600);
    const d1 = await dist(p), s1 = await screenOf(p, S1);
    out.push({ name: 'pc 球面 ホイールで寄る（ポインタの下の点は留まる、3px 以内）', ok: d1 < d0 * 0.6 && off(s1, [px, py]) < 3, info: `距離 ${d0.toFixed(2)}→${d1.toFixed(2)} ずれ=${off(s1, [px, py]).toFixed(1)}px` });
    // 寄った状態では同じドラッグで回る角が小さい
    const e2 = await eyeDir(p), Sc2 = await surf(p, cx, cy);
    await drag(p, cx, cy, 80, 30);
    const a2 = angle(e2, await eyeDir(p)) * 0.5, sc2 = await screenOf(p, Sc2);  // 80px → 40px あたりに換算
    out.push({ name: 'pc 球面 寄った状態でも面の点がポインタに付いてくる（3px 以内）', ok: off(sc2, [cx + 80, cy + 30]) < 3, info: `ずれ=${off(sc2, [cx + 80, cy + 30]).toFixed(1)}px` });
    out.push({ name: 'pc 球面 寄ると同じドラッグの回転角が小さい', ok: a2 < a1 * 0.6, info: `全体 ${(a1 * 180 / Math.PI).toFixed(1)}° → 寄り ${(a2 * 180 / Math.PI).toFixed(1)}°` });
    // ホイールで離れる（点は留まる）
    const S2 = await surf(p, px, py);
    await p.mouse.move(px, py);
    for (let i = 0; i < 4; i++) { await p.mouse.wheel({ deltaY: 100 }); await lib.sleep(60); }
    await lib.sleep(600);
    const d2 = await dist(p), s2 = S2 && await screenOf(p, S2);
    out.push({ name: 'pc 球面 ホイールで離れる（点は留まる）', ok: d2 > d1 * 1.5 && !!S2 && off(s2, [px, py]) < 3, info: `距離 ${d1.toFixed(2)}→${d2.toFixed(2)}` + (S2 ? ` ずれ=${off(s2, [px, py]).toFixed(1)}px` : '') });
    // 離れ過ぎない
    for (let i = 0; i < 30; i++) { await p.mouse.wheel({ deltaY: 300 }); await lib.sleep(20); }
    await lib.sleep(600);
    const d3 = await dist(p);
    out.push({ name: 'pc 球面 離れる上限（読み込み時の 4 倍）', ok: d3 <= d0 * 4 + 1e-6 && d3 > d0 * 3.5, info: `距離=${d3.toFixed(2)} (全体 ${d0.toFixed(2)})` });
    // ダブルクリックで全体に戻る
    await p.mouse.move(cx, cy); await p.mouse.click(cx, cy, { count: 2 }); await lib.sleep(1600);  // clickCount=2 で DOM の dblclick が出る（1 を 2 回では出ない）
    const d4 = await dist(p);
    out.push({ name: 'pc 球面 ダブルクリックで全体に戻る', ok: Math.abs(d4 - d0) < 1e-3, info: `距離=${d4.toFixed(2)}` });
    const errs = await p.evaluate(() => []).then(() => p._errors);
    out.push({ name: 'pc 球面 ページエラーなし', ok: errs.length === 0, info: errs.slice(0, 2).join(' | ') });
    await p.screenshot({ path: `${lib.OUT}/globe_camera_pc.png` });
    await p.close();
  }

  // ---- スマホ: ピンチは指の間の面の点を留める ----
  {
    const p = await lib.open(browser, url('/globe/'), 'phone');
    await lib.plotted(p); await lib.sleep(1500);
    const [cx, cy] = await center(p);
    const d0 = await dist(p);
    const Sc = await surf(p, cx, cy + 40);
    await p.evaluate(async (cx, cy) => {  // 合成 TouchEvent（CDP のタッチは時刻列が実機とかけ離れる）
      const cv = plot.querySelector('canvas');
      const mk = (id, x, y) => new Touch({ identifier: id, target: cv, clientX: x, clientY: y, pageX: x, pageY: y, radiusX: 2, radiusY: 2, force: 1 });
      const fire = (type, ts) => cv.dispatchEvent(new TouchEvent(type, { touches: ts, targetTouches: ts, changedTouches: ts, bubbles: true, cancelable: true }));
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      fire('touchstart', [mk(1, cx - 40, cy + 40), mk(2, cx + 40, cy + 40)]);
      for (let i = 1; i <= 10; i++) { const w = 40 + 8 * i; fire('touchmove', [mk(1, cx - w, cy + 40), mk(2, cx + w, cy + 40)]); await sleep(30); }
      await sleep(200);
      fire('touchend', []);
    }, cx, cy);
    await lib.sleep(600);
    const d1 = await dist(p), s1 = await screenOf(p, Sc);
    out.push({ name: 'phone 球面 ピンチで寄る（指の間の面の点は留まる、4px 以内）', ok: d1 < d0 * 0.8 && off(s1, [cx, cy + 40]) < 4, info: `距離 ${d0.toFixed(2)}→${d1.toFixed(2)} ずれ=${off(s1, [cx, cy + 40]).toFixed(1)}px` });
    await p.screenshot({ path: `${lib.OUT}/globe_camera_phone.png` });
    await p.close();
  }
  return out;
};
