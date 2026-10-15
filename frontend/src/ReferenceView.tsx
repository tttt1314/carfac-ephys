import Plot from './Plot';
import { NORMAL, IMPAIRED, REFERENCE, FREQUENCY_TICKS, PanelHeading } from './ExploreView';
import type { BenchmarkResult, Reference } from './types';

export default function ReferenceView({ reference, loading, error, retry, benchmark }: { reference: Reference | null; loading: boolean; error: string; retry: () => void; benchmark: BenchmarkResult | null }) {
  if (loading) return <div className="reference-loading"><span className="spinner" /><p>讀取實測參考資料…</p></div>;
  if (error) return <div className="error-box" role="alert"><strong>參考資料讀取失敗</strong><p>{error}</p><button className="secondary-button" onClick={retry}>重新讀取</button></div>;
  if (!reference) return null;
  const observed = [...reference.points].sort((a, b) => a.dp_hz - b.dp_hz);
  const limitations = new Set(reference.limitations.map(text => text.trim()));
  const additionalWarnings = benchmark?.warnings.filter(text => !limitations.has(text.trim())) ?? [];
  const license = typeof reference.license === 'string' ? reference.license : reference.license.label ?? reference.license.name ?? '詳見來源';
  const refTrace = { type: 'scatter', mode: 'lines', x: observed.map(p => p.dp_hz), y: observed.map(p => p.level_db_spl), name: '單耳實測', line: { color: REFERENCE, width: 1.5 }, hovertemplate: 'DP %{x:.1f} Hz<br>%{y:.2f} dB SPL<extra>單耳實測</extra>' };
  const x = benchmark?.dp_hz ?? [];
  const relativeRefX = benchmark && benchmark.reference_relative_db.length === x.length ? x : observed.map(p => p.dp_hz);
  return <>
    <section className="reference-intro"><span className="eyebrow">OBSERVED DATA FIRST</span><h2>先看實測，再比較模型。</h2><p>{reference.title}</p><div className="reference-tags"><span>單耳實測（聽力狀態未註明）</span><span>{reference.component}</span><a href={reference.source_url} target="_blank" rel="noreferrer">查看原始資料 ↗</a></div><div className="reference-facts"><div><span>Primary ratio</span><strong>f₂ / f₁ = {reference.stimulus.ratio}</strong></div><div><span>Measured primaries</span><strong>{reference.stimulus.l1_db_fpl} / {reference.stimulus.l2_db_fpl} dB FPL</strong></div><div><span>Observed points</span><strong>{reference.points.length} points</strong></div><div><span>License</span><strong>{license}</strong></div></div></section>

    <section className="plot-panel"><PanelHeading number="01" title="原始實測振幅曲線" english="Observed DPOAE magnitude"><span className="badge">未經模型擬合</span></PanelHeading><Plot label="原始單耳 DPOAE 參考曲線，單位 dB SPL" height={320} data={[refTrace]} layout={{ showlegend: false, xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency · 2f₁ − f₂ (Hz)' }, gridcolor: '#edf1f3' }, yaxis: { title: { text: 'Measured DPOAE level (dB SPL)' }, gridcolor: '#edf1f3' } }} /><p className="panel-explanation">橫軸是輸出 DP frequency，並非 f₂ 或 cochlear CF。圖形直接來自參考資料；此曲線不能被當成正常聽力的群體平均。</p></section>

    {!benchmark && <section className="reference-callout"><div><span className="panel-number">02</span><h3>建立可追蹤的參考比較</h3><p>左側選擇 OHC 參數與中耳濾波，按「執行參考比較」，產生 24 點正常／修改模型曲線。</p></div><p>兩條模型曲線共用正常模型 anchor；實測減去自身第一點。實測 dB SPL 與模型 dB re 1 分開呈現，不會直接疊成相同單位。</p></section>}

    {benchmark && <>
      <section className="plot-panel"><PanelHeading number="02" title="相對振幅趨勢" english="Relative magnitude comparison"><span className="badge">24-point comparison</span></PanelHeading><Plot label="實測與正常及修改模型相對振幅趨勢" height={330} data={[
        { type: 'scatter', mode: 'lines', x: relativeRefX, y: benchmark.reference_relative_db, name: '單耳實測 · relative', line: { color: REFERENCE, width: 1.5 } },
        { type: 'scatter', mode: 'lines+markers', x, y: benchmark.normal_relative_db, name: '正常模型 · relative', line: { color: NORMAL, width: 2 }, marker: { size: 4 } },
        { type: 'scatter', mode: 'lines+markers', x, y: benchmark.impaired_relative_db, name: '修改模型 · relative', line: { color: IMPAIRED, width: 2 }, marker: { size: 4 } },
      ]} layout={{ xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency (Hz)' }, gridcolor: '#edf1f3' }, yaxis: { title: { text: 'Relative level (dB)' }, gridcolor: '#edf1f3' }, hovermode: 'x unified' }} /><p className="panel-explanation">這是頻率趨勢比較，不代表模型已完成絕對聲壓校準。兩條模型曲線共用正常模型 anchor，保留兩者振幅差異；實測曲線減去自身第一點。修改模型沒有各自正規化。</p></section>

      <section className="plot-panel"><PanelHeading number="03" title="保留原始單位" english="Separate absolute scales" /><div className="waveform-grid"><div><h3>實測 · dB SPL</h3><Plot label="實測原始 dB SPL 振幅" height={260} data={[refTrace]} layout={{ showlegend: false, xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency (Hz)' } }, yaxis: { title: { text: 'DPOAE level (dB SPL)' } } }} /></div><div><h3>模型 · dB re 1 model unit</h3><Plot label="正常和修改模型的未校準輸出振幅" height={260} data={[{ type: 'scatter', mode: 'lines+markers', x, y: benchmark.normal_db, name: '正常模型', line: { color: NORMAL, width: 1.8 }, marker: { size: 4 } }, { type: 'scatter', mode: 'lines+markers', x, y: benchmark.impaired_db, name: '修改模型', line: { color: IMPAIRED, width: 1.8 }, marker: { size: 4 } }]} layout={{ xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency (Hz)' } }, yaxis: { title: { text: 'Level (dB re 1 model unit)' } } }} /></div></div></section>

      <details className="input-details"><summary>Phase · 保留各自的參考定義 <span>Exploratory, no absolute alignment</span></summary><div className="waveform-grid"><div><h3>實測原始 phase · unwrapped cycles</h3><Plot label="單耳實測原始 unwrapped phase cycles" height={270} data={[{ type: 'scatter', mode: 'lines', x: observed.map(p => p.dp_hz), y: observed.map(p => p.phase_cycles), name: '單耳實測 · raw phase', line: { color: REFERENCE, width: 1.3 } }]} layout={{ showlegend: false, xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency (Hz)' } }, yaxis: { title: { text: 'Measured raw phase (cycles)' } } }} /></div><div><h3>模型 cosine-reference phase · wrapped cycles</h3><Plot label="正常與修改模型 cosine-reference phase，wrapped 在負零點五到零點五 cycles" height={270} data={[{ type: 'scatter', mode: 'markers', connectgaps: false, x, y: benchmark.normal_phase_cycles, name: '正常模型', marker: { color: NORMAL, size: 5 } }, { type: 'scatter', mode: 'markers', connectgaps: false, x, y: benchmark.impaired_phase_cycles, name: '修改模型', marker: { color: IMPAIRED, size: 5, symbol: 'diamond-open' } }]} layout={{ xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'DP frequency (Hz)' } }, yaxis: { title: { text: 'Model cosine-reference phase (cycles)' }, range: [-.5, .5], tickvals: [-.5, -.25, 0, .25, .5] } }} /></div></div><p className="panel-explanation">兩側保留不同的 phase reference，沒有絕對對齊。模型只顯示 cosine-reference 的 wrapped phase（−0.5 至 0.5 cycles），不對稀疏刺激點自動 unwrap；低於振幅底限的 phase 留空。這裡不估計 latency，也不判定 reverse path 正確。</p></details>
      <div className="benchmark-foot">本次比較運算時間 <strong>{benchmark.compute_seconds.toFixed(1)} s</strong> · 24 個模型刺激點</div>
    </>}

    <section className="limitations-panel"><h3>證據範圍與比較限制</h3><ul>{reference.limitations.map((limitation, i) => <li key={i}>{limitation}</li>)}<li>Measured FPL 與模型 nominal SPL 僅作近似刺激對照；前向模型、逆向耦合與中耳路徑仍需要獨立驗證。</li></ul><p>來源 DOI：<a href={reference.doi.startsWith('http') ? reference.doi : `https://doi.org/${reference.doi}`} target="_blank" rel="noreferrer">{reference.doi}</a></p></section>
    {additionalWarnings.length > 0 && <div className="warnings"><strong>本次比較備註</strong><ul>{additionalWarnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul></div>}
  </>;
}
