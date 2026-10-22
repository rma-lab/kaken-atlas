// 拡大に応じた点の大きさ（3 ビュー共通、連続）: 全体表示は基準、4 倍で直径 ×4^0.3≒1.52（2D は不透明度も上がる）、40 倍は上限、戻すと基準。
// 2D は Plotly が範囲変更のたびに点の設定を再適用しても大きさが保たれる。球面は自前カメラのズーム（ホイール）でも同じフレームで追随する。
// evaluate から Plotly の promise をそのまま返さない（plot 要素が返り、直列化で固まる）
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const st = (p) => p.evaluate(() => plot._dbgPointSize());
  const near = (a, b, tol) => Math.abs(a - b) <= tol;
  {
    const p = await lib.open(browser, url('/map2d/'), 'pc');
    await lib.plotted(p); await lib.sleep(1500);
    const s0 = await st(p);
    out.push({ name: '2D 全体表示は基準の大きさ', ok: s0.scale === 1 && s0.applied.size === s0.base.size, info: `size=${s0.applied.size}` });
    const zoom = async (z) => { await p.evaluate((z) => { const w = (homeRange.x[1] - homeRange.x[0]) / z, cx = 3.5, cy = 7.5; const hy = w / 2 * plot._fullLayout.yaxis._length / plot._fullLayout.xaxis._length; return Plotly.relayout(plot, { 'xaxis.range': [cx - w / 2, cx + w / 2], 'yaxis.range': [cy - hy, cy + hy] }).then(() => 1); }, z); await lib.sleep(400); };
    await zoom(4);
    const s1 = await st(p);
    out.push({ name: '2D 4 倍で直径 ×1.52・不透明度 0.73', ok: near(s1.applied.size, s0.base.size * Math.pow(4, 0.3), 0.03) && near(s1.applied.opacity, 0.733, 0.01), info: `size=${s1.applied.size} opacity=${s1.applied.opacity}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range[0]': plot._fullLayout.xaxis.range[0] - 0.05, 'xaxis.range[1]': plot._fullLayout.xaxis.range[1] - 0.05 }).then(() => 1)); await lib.sleep(400);
    const s1b = await st(p);
    out.push({ name: '2D 平行移動後も大きさが保たれる', ok: near(s1b.applied.size, s1.applied.size, 0.01), info: `size=${s1b.applied.size}` });
    await zoom(40);
    const s2 = await st(p);
    out.push({ name: '2D 40 倍は上限 5px・不透明度 0.85', ok: s2.applied.size === 5 && s2.applied.opacity === 0.85, info: `size=${s2.applied.size} opacity=${s2.applied.opacity}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': homeRange.x, 'yaxis.range': homeRange.y }).then(() => 1)); await lib.sleep(400);
    const s3 = await st(p);
    out.push({ name: '2D 全体に戻すと基準に戻る', ok: s3.scale === 1 && s3.applied.size === s0.base.size && near(s3.applied.opacity, s0.base.opacity, 0.001), info: `size=${s3.applied.size} opacity=${s3.applied.opacity}` });
    out.push({ name: '2D ページエラーなし', ok: p._errors.length === 0, info: p._errors.slice(0, 2).join(' | ') });
    await p.close();
  }
  for (const view of ['map3d', 'globe']) {
    const p = await lib.open(browser, url(`/${view}/`), 'pc');
    await lib.plotted(p); await lib.sleep(1500);
    const s0 = await st(p);
    out.push({ name: `${view} 全体表示は基準の大きさ`, ok: s0.scale === 1 && s0.applied.pixelRatio === s0.applied.basePR, info: JSON.stringify(s0.applied) });
    await p.evaluate(() => { const c = liveCamera(), R = surfaceR(); const d = Math.hypot(c.eye.x - c.ctr.x, c.eye.y - c.ctr.y, c.eye.z - c.ctr.z);
      const k = (R + (d - R) / 4) / d;  // 面（注視点）までの距離を 1/4 に＝4 倍
      return Plotly.relayout(plot, { 'scene.camera.eye': { x: c.ctr.x + (c.eye.x - c.ctr.x) * k, y: c.ctr.y + (c.eye.y - c.ctr.y) * k, z: c.ctr.z + (c.eye.z - c.ctr.z) * k } }).then(() => 1); });
    await lib.sleep(500);
    const s1 = await st(p);
    out.push({ name: `${view} 4 倍で pixelRatio ×1.52（全点オブジェクトを包んだ）`, ok: near(s1.applied.pixelRatio / s1.applied.basePR, Math.pow(4, 0.3), 0.02) && s1.applied.wrapped === s1.applied.n, info: `wrapped=${s1.applied.wrapped}/${s1.applied.n} ` + `zoom=${s1.zoom} ratio=${(s1.applied.pixelRatio / s1.applied.basePR).toFixed(3)}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'scene.camera.eye': homeCam.eye, 'scene.camera.center': homeCam.ctr, 'scene.camera.up': homeCam.up }).then(() => 1)); await lib.sleep(500);
    const s2 = await st(p);
    out.push({ name: `${view} 全体に戻すと基準に戻る`, ok: s2.scale === 1 && s2.applied.pixelRatio === s2.applied.basePR });
    if (view === 'globe') {  // 自前カメラ（ホイール相当）でも追随
      const [cx, cy] = await p.evaluate(() => { const r = plot.querySelector('canvas').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; });
      await p.mouse.move(cx, cy);
      for (let i = 0; i < 6; i++) { await p.mouse.wheel({ deltaY: -100 }); await lib.sleep(50); }
      await lib.sleep(300);
      const s3 = await st(p);
      out.push({ name: 'globe ホイールのズームにも追随', ok: s3.zoom > 2 && near(s3.applied.pixelRatio / s3.applied.basePR, Math.min(3.5 / s3.base.size, Math.pow(s3.zoom, 0.3)), 0.03), info: `zoom=${s3.zoom} ratio=${(s3.applied.pixelRatio / s3.applied.basePR).toFixed(3)}` });
    }
    out.push({ name: `${view} ページエラーなし`, ok: p._errors.length === 0, info: p._errors.slice(0, 2).join(' | ') });
    await p.close();
  }
  return out;
};
