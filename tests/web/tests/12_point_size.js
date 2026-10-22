// 拡大に応じた点の大きさ（3 ビュー共通）: 全体表示は基準の大きさ、寄ると段階的に大きく濃く（ズームが止まってから 1 回 restyle）、戻すと基準に戻る。
// 1.5 倍未満の範囲変更では restyle しない（07_perf の前提）。evaluate から Plotly の promise をそのまま返さない（plot 要素が返り、直列化で固まる）
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const st = (p) => p.evaluate(() => plot._dbgPointSize());
  // 2D
  {
    const p = await lib.open(browser, url('/map2d/'), 'pc');
    await lib.plotted(p); await lib.sleep(1500);
    const s0 = await st(p);
    out.push({ name: '2D 全体表示は基準の大きさ', ok: s0.level === 0 && s0.size === s0.base.size, info: `size=${s0.size} opacity=${s0.opacity}` });
    const zoom = async (z) => { await p.evaluate((z) => { const w = (homeRange.x[1] - homeRange.x[0]) / z, cx = 3.5, cy = 7.5; const hy = w / 2 * plot._fullLayout.yaxis._length / plot._fullLayout.xaxis._length; return Plotly.relayout(plot, { 'xaxis.range': [cx - w / 2, cx + w / 2], 'yaxis.range': [cy - hy, cy + hy] }).then(() => 1); }, z); await lib.sleep(900); };
    await zoom(1.3);
    const s1 = await st(p);
    out.push({ name: '2D 1.3 倍では変えない', ok: s1.level === 0 && s1.size === s0.size, info: `zoom=${s1.zoom}` });
    await zoom(4);
    const s2 = await st(p);
    out.push({ name: '2D 4 倍で大きく濃く', ok: s2.level === 3 && s2.size > s0.size * 1.5 && s2.opacity > s0.opacity, info: `size=${s2.size} opacity=${s2.opacity}` });
    await zoom(40);
    const s3 = await st(p);
    out.push({ name: '2D 40 倍は上限 7px', ok: s3.size <= 7 && s3.size >= 6 && s3.opacity === 0.9, info: `level=${s3.level} size=${s3.size}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': homeRange.x, 'yaxis.range': homeRange.y }).then(() => 1)); await lib.sleep(900);
    const s4 = await st(p);
    out.push({ name: '2D 全体に戻すと基準に戻る', ok: s4.level === 0 && s4.size === s0.size && s4.opacity === s0.opacity });
    out.push({ name: '2D ページエラーなし', ok: p._errors.length === 0, info: p._errors.slice(0, 2).join(' | ') });
    await p.close();
  }
  // 3D と球面
  for (const view of ['map3d', 'globe']) {
    const p = await lib.open(browser, url(`/${view}/`), 'pc');
    await lib.plotted(p); await lib.sleep(1500);
    const s0 = await st(p);
    out.push({ name: `${view} 全体表示は基準の大きさ`, ok: s0.level === 0 && s0.size === s0.base.size, info: `size=${s0.size} opacity=${s0.opacity}` });
    await p.evaluate(() => { const c = liveCamera(), R = surfaceR(); const d = Math.hypot(c.eye.x - c.ctr.x, c.eye.y - c.ctr.y, c.eye.z - c.ctr.z);
      const k = (R + (d - R) / 4) / d;  // 面（注視点）までの距離を 1/4 に＝4 倍
      return Plotly.relayout(plot, { 'scene.camera.eye': { x: c.ctr.x + (c.eye.x - c.ctr.x) * k, y: c.ctr.y + (c.eye.y - c.ctr.y) * k, z: c.ctr.z + (c.eye.z - c.ctr.z) * k } }).then(() => 1); });
    await lib.sleep(1200);
    const s1 = await st(p);
    out.push({ name: `${view} 4 倍で大きく濃く`, ok: s1.level === 3 && s1.size > s0.size * 1.5 && s1.opacity > s0.opacity, info: `zoom=${s1.zoom} size=${s1.size} opacity=${s1.opacity}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'scene.camera.eye': homeCam.eye, 'scene.camera.center': homeCam.ctr, 'scene.camera.up': homeCam.up }).then(() => 1)); await lib.sleep(1200);
    const s2 = await st(p);
    out.push({ name: `${view} 全体に戻すと基準に戻る`, ok: s2.level === 0 && s2.size === s0.size && s2.opacity === s0.opacity });
    out.push({ name: `${view} ページエラーなし`, ok: p._errors.length === 0, info: p._errors.slice(0, 2).join(' | ') });
    await p.close();
  }
  return out;
};
