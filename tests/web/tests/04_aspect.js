// 2D の縦横比（scaleanchor なしの自前管理）: 初期表示・歪んだ範囲指定・フライトゥ・リサイズ・ダブルクリック
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  const eq = (a) => Math.abs(a.ratio - 1) < 0.002;
  for (const device of ['phone', 'pc']) {
    const p = await lib.open(browser, url('/map2d/'), device);
    await lib.ready(p); await lib.sleep(1200);
    const a0 = await lib.aspect(p);
    const ext = await p.evaluate(() => { let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; for (let i = 0; i < M.n; i += 7) { x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]); y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]); } return [x0, x1, y0, y1]; });
    const fits = a0.xr[0] <= ext[0] && a0.xr[1] >= ext[1] && a0.yr[0] <= ext[2] && a0.yr[1] >= ext[3];
    out.push({ name: `${device} 初期表示が等スケールで全体を含む`, ok: eq(a0) && fits, info: `ratio=${a0.ratio}` });
    await p.evaluate(() => Plotly.relayout(plot, { 'xaxis.range': [2, 6], 'yaxis.range': [2, 3] })); await lib.sleep(400);
    out.push({ name: `${device} 歪んだ範囲指定が補正される`, ok: eq(await lib.aspect(p)) });
    await lib.searchAndOpenFirst(p, device, '重力波');
    const a2 = await lib.aspect(p);
    out.push({ name: `${device} フライトゥ後（幅 5.0）`, ok: eq(a2) && Math.abs((a2.xr[1] - a2.xr[0]) - 5) < 0.05, info: `width=${(a2.xr[1] - a2.xr[0]).toFixed(2)}` });
    const vp = p.viewport(); await p.setViewport(Object.assign({}, vp, { width: vp.height, height: vp.width })); await lib.sleep(1200);
    out.push({ name: `${device} リサイズ後`, ok: eq(await lib.aspect(p)) });
    if (device === 'pc') {
      await p.mouse.move(300, 400); await p.mouse.click(300, 400); await lib.sleep(120); await p.mouse.click(300, 400); await lib.sleep(1500);
      const a4 = await lib.aspect(p);  // リサイズ後なので初期範囲とは一致しない。等スケールで全体を含めば良い
      const fits4 = a4.xr[0] <= ext[0] && a4.xr[1] >= ext[1] && a4.yr[0] <= ext[2] && a4.yr[1] >= ext[3];
      const width4 = a4.xr[1] - a4.xr[0];
      out.push({ name: 'pc ダブルクリックで全体に戻る', ok: eq(a4) && fits4 && width4 > 10, info: `xr=${JSON.stringify(a4.xr)} (フライトゥ後は幅 5)` });
    }
    await p.close();
  }
  return out;
};
