// 3 ビューが開き、点データ・色データ・共有シャードが整合しているか
module.exports = async ({ browser, url, lib }) => {
  const out = [];
  for (const view of ['map2d', 'map3d', 'globe']) {
    const p = await lib.open(browser, url(`/${view}/`), 'pc');
    await lib.ready(p); await lib.sleep(500);
    const r = await p.evaluate(() => {
      let bad = 0, seen = 0;
      for (let g = 0; g < M.n; g++) { const row = getRow(g); if (!row) continue; seen++; const tr = M.traces[traceOf[g]]; if (tr.cat && tr.cat !== M.cats[row[4]]) bad++; }
      let badColor = 0; for (let i = 0; i < M.n; i += 997) if (!/^rgb\(\d+,\d+,\d+\)$/.test(colorOf[i])) badColor++;
      return { n: M.n, seen, bad, traces: M.traces.length, colors: colorOf.length, badColor, sectors: M.colorLegend ? M.colorLegend.sectors.length : 0 };
    });
    out.push({ name: `${view} 読み込み`, ok: r.n === 206078 && p._errors.length === 0, info: `n=${r.n} traces=${r.traces} errors=${p._errors.length}` });
    out.push({ name: `${view} シャード行の種目がトレースと一致`, ok: r.seen === r.n && r.bad === 0, info: `seen=${r.seen} bad=${r.bad}` });
    out.push({ name: `${view} 点ごとの色`, ok: r.colors === r.n && r.badColor === 0 && r.sectors === 12, info: `colors=${r.colors} sectors=${r.sectors}` });
    out.push({ name: `${view} トレース数 ≤ 255（3D 判定の上限）`, ok: r.traces + 2 <= 255, info: `${r.traces}` });
    await p.close();
  }
  return out;
};
