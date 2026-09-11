// 2D の範囲変更（ピンチ・パン相当）の所要時間。scaleanchor の罠の再発防止（v1.3 で 190ms → 33ms）
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  for (const device of ['phone', 'pc']) {
    const p = await lib.open(browser, url('/map2d/'), device);
    await lib.plotted(p); await lib.sleep(3000);
    const med = await p.evaluate(async () => {
      const xa = plot._fullLayout.xaxis, ya = plot._fullLayout.yaxis, xr = xa.range.slice(), yr = ya.range.slice(), t = [];
      for (let i = 0; i < 12; i++) { const s = 0.7 + 0.03 * i, cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2; const t1 = performance.now();
        await Plotly.relayout(plot, { 'xaxis.range': [cx - (xr[1] - xr[0]) * s / 2, cx + (xr[1] - xr[0]) * s / 2], 'yaxis.range': [cy - (yr[1] - yr[0]) * s / 2, cy + (yr[1] - yr[0]) * s / 2] });
        await new Promise((r) => requestAnimationFrame(r)); t.push(performance.now() - t1); }
      t.sort((a, b) => a - b); return t[6];
    });
    out.push({ name: `${device} 2D 範囲変更の中央値 < 80ms`, ok: med < 80, info: `${med.toFixed(1)}ms` });
    await p.close();
  }
  return out;
};
