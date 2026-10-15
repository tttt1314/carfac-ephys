import { useEffect, useState } from 'react';
import Plot from './Plot';
import type { Trace } from './Plot';
import type { Condition, ExploreResult, SourceComponent } from './types';
import { commonBounds, formatHz } from './model';

export const NORMAL = '#087f8c';
export const IMPAIRED = '#c58b2a';
export const REFERENCE = '#3e5066';
export const FREQUENCY_TICKS = {
  tickvals: [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
  ticktext: ['5', '10', '20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k'],
};
const color = (id: string) => id === 'normal' ? NORMAL : IMPAIRED;
const label = (id: string) => id === 'normal' ? '正常模型' : '修改模型';
const heatScale = [[0, '#f4f7fa'], [.25, '#d5e6eb'], [.5, '#82bac3'], [.75, '#2f8595'], [1, '#123d59']];
const sourceLabels: Record<SourceComponent, string> = { total: 'Total · fast + AGC + gain', fast: 'Fast · OHC 快速非線性', agc: 'AGC · 控制變化', gain: 'Gain · stage gain 變化' };

export function PanelHeading({ number, title, english, children }: { number: string; title: string; english: string; children?: React.ReactNode }) {
  return <div className="panel-heading"><div className="panel-title"><span className="panel-number">{number}</span><h2>{title}<span>{english}</span></h2></div>{children}</div>;
}

export function EmptyExplore() {
  return <>
    <section className="welcome-panel"><span className="eyebrow">FOLLOW THE SIGNAL</span><h2>讓每一步耳蝸反應，都看得見。</h2><p>選擇雙音或音訊，並排比較正常與修改 OHC 參數後的模型。圖表只在實際計算完成後顯示。</p><div className="signal-flow"><div><span>01 / INPUT</span><strong>音訊刺激</strong><small>波形與輸入頻譜</small></div><b aria-hidden="true">→</b><div><span>02 / COCHLEA</span><strong>CARFAC 反應</strong><small>BM 與非線性 source</small></div><b aria-hidden="true">→</b><div><span>03 / EMISSION</span><strong>逆向模型輸出</strong><small>完整頻譜，最後才分析 DP</small></div></div><p className="welcome-foot">← 從左側設定開始，按「執行 CARFAC」。首次運算需要編譯模型。</p></section>
    <div className="empty-grid"><section className="empty-card"><span className="panel-number">01</span><h3>BM cochleagram</h3><p>時間 × cochlear CF。觀察刺激能量在哪些 channels 出現，以及反應如何隨時間變化。</p><span className="awaiting">等待實際結果</span></section><section className="empty-card"><span className="panel-number">02</span><h3>Nonlinear source</h3><p>比較 fast、AGC、gain 的時域 residual；包含非線性造成的基頻變化與新頻率。</p><span className="awaiting">等待實際結果</span></section><section className="empty-card"><span className="panel-number">03</span><h3>Output spectrum</h3><p>兩個條件使用相同分析窗。雙音會標出 f₁、f₂ 與 2f₁−f₂；音訊不套用 DP 標記。</p><span className="awaiting">等待實際結果</span></section></div>
  </>;
}

function HeatmapPair({ result, kind, component, selected, onSelect }: { result: ExploreResult; kind: 'bm' | 'source'; component: SourceComponent; selected: number; onSelect: (index: number) => void }) {
  const matrices = result.conditions.map(condition => kind === 'bm' ? condition.bm.rms_db : condition.sources[`${component}_db`]);
  const [low, high] = commonBounds(matrices);
  const nearest = (hz: number) => onSelect(result.cfs_hz.reduce((best, cf, i) => Math.abs(Math.log(cf / hz)) < Math.abs(Math.log(result.cfs_hz[best] / hz)) ? i : best, 0));
  return <><div className="heatmap-pair">{result.conditions.map((condition, i) => <div className="condition-plot" key={condition.id}><div className="condition-label"><i className={`dot ${condition.id}`} /><strong>{label(condition.id)}</strong><span>{condition.id === 'normal' ? 'Health = 1' : 'OHC 參數已修改'}</span></div><Plot label={`${label(condition.id)} ${kind === 'bm' ? 'BM' : component + ' source'} cochleagram`} height={300} onChannelClick={nearest} data={[{ type: 'heatmap', x: (kind === 'bm' ? condition.bm.time_s : condition.sources.time_s).map(t => t * 1000), y: result.cfs_hz, z: matrices[i], colorscale: heatScale, zmin: low, zmax: high, showscale: true, colorbar: { thickness: 9, len: .88, ticksuffix: ' dB', tickfont: { size: 10 }, outlinewidth: 0 }, hovertemplate: 'Time %{x:.1f} ms<br>CF %{y:.0f} Hz<br>%{z:.1f} dB re 1 model unit<extra></extra>' }]} layout={{ margin: { l: 60, r: 65, t: 10, b: 45 }, xaxis: { title: { text: 'Time (ms)' }, gridcolor: '#edf1f3' }, yaxis: { type: 'log', title: { text: 'Cochlear CF (Hz)' }, tickvals: [125, 250, 500, 1000, 2000, 4000, 8000, 16000], ticktext: ['125', '250', '500', '1k', '2k', '4k', '8k', '16k'] }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: result.cfs_hz[selected], y1: result.cfs_hz[selected], line: { color: '#b88120', width: 1, dash: 'dot' } }] }} /></div>)}</div><p className="chart-caption">共用色階 {low} 至 {high} dB re 1 model unit（{high - low} dB 範圍）；低於下界的值以最淡色顯示。點選 CF 可檢查波形；CF 為模型 pole frequency。</p></>;
}

function spectrumTrace(condition: Condition): Trace {
  const valid = condition.spectrum.frequency_hz.map((f, i) => ({ f, level: condition.spectrum.level_db[i] })).filter(p => p.f > 0);
  return { x: valid.map(p => p.f), y: valid.map(p => p.level), type: 'scatter', mode: 'lines', name: label(condition.id), line: { color: color(condition.id), width: 1.8 }, hovertemplate: '%{x:.1f} Hz<br>%{y:.2f} dB re 1 model unit<extra>%{fullData.name}</extra>' };
}

export default function ExploreView({ result }: { result: ExploreResult }) {
  const [component, setComponent] = useState<SourceComponent>('total');
  const [selected, setSelected] = useState(0);
  useEffect(() => {
    setSelected(result.cfs_hz.reduce((best, cf, i) => Math.abs(cf - 1000) < Math.abs(result.cfs_hz[best] - 1000) ? i : best, 0));
  }, [result]);
  const traceStart = result.conditions[0]?.waveforms.time_s[0] ?? result.analysis_window_s[0];
  const markerItems = result.markers ? [{ hz: result.markers.f1_hz, name: 'f₁', color: '#91a4af' }, { hz: result.markers.f2_hz, name: 'f₂', color: '#91a4af' }, { hz: result.markers.dp_hz, name: '2f₁ − f₂', color: '#aa5f66' }].filter(p => p.hz > 0 && p.hz < result.sample_rate / 2) : [];
  const waveformTraces = (kind: 'bm' | 'source'): Trace[] => result.conditions.map(condition => {
    const signal = kind === 'bm' ? condition.waveforms.bm : condition.waveforms[`source_${component}`];
    const samples = condition.waveforms.time_s.map((t, i) => ({ t, value: signal[selected]?.[i] })).filter(p => p.t >= traceStart && p.t <= traceStart + .02);
    return { type: 'scatter', mode: 'lines', x: samples.map(p => p.t * 1000), y: samples.map(p => p.value), name: label(condition.id), line: { color: color(condition.id), width: 1.4 }, hovertemplate: '%{x:.3f} ms<br>%{y:.5g} model unit<extra>%{fullData.name}</extra>' };
  });
  const normalDP = result.conditions.find(c => c.id === 'normal')?.spectrum.dp_level_db;
  const impairedDP = result.conditions.find(c => c.id === 'impaired')?.spectrum.dp_level_db;
  return <>
    <section className="results-strip" aria-label="本次計算摘要"><div><span>ANALYSIS WINDOW</span><strong>{(result.analysis_window_s[0] * 1000).toFixed(0)}–{(result.analysis_window_s[1] * 1000).toFixed(0)} <small>ms</small></strong></div><div><span>COCHLEAR CHANNELS</span><strong>{result.cfs_hz.length} <small>channels</small></strong></div><div><span>COMPUTE TIME</span><strong>{result.compute_seconds.toFixed(1)} <small>s</small></strong></div>{result.markers && <div><span>2f₁ − f₂</span><strong>{formatHz(result.markers.dp_hz)}{typeof normalDP === 'number' && typeof impairedDP === 'number' && <small className="dp-delta">Δ {(impairedDP - normalDP).toFixed(1)} dB</small>}</strong></div>}</section>

    <section className="plot-panel"><PanelHeading number="01" title="基底膜反應" english="BM cochleagram"><span className="badge">5 ms block RMS</span></PanelHeading><HeatmapPair result={result} kind="bm" component={component} selected={selected} onSelect={setSelected} /><p className="panel-explanation">CAR output 經 AC coupling 後的模型反應。以 5 ms 不重疊區塊的 RMS 顯示局部振幅；不是校準的 BM 位移或速度，也不是 FFT spectrogram。</p></section>

    <section className="plot-panel"><PanelHeading number="02" title="非線性來源" english="Output-equivalent source"><label className="compact-field"><span className="sr-only">Source component</span><select aria-label="Source component" value={component} onChange={e => setComponent(e.target.value as SourceComponent)}>{Object.entries(sourceLabels).map(([value, title]) => <option key={value} value={value}>{title}</option>)}</select></label></PanelHeading><HeatmapPair result={result} kind="source" component={component} selected={selected} onSelect={setSelected} /><p className="panel-explanation">相對模型自身無聲線性化的時域 residual。包含基頻壓縮、harmonics 與 intermodulation；數值是輸出等效模型量，並非 OHC 力或單獨的 DP。</p></section>

    <section className="plot-panel"><PanelHeading number="03" title="逆向模型輸出頻譜" english="Output spectrum"><span className="unit-label">dB re 1 RMS model unit</span></PanelHeading><Plot label="正常與修改模型的逆向輸出頻譜" height={310} data={result.conditions.map(spectrumTrace)} layout={{ hovermode: 'x unified', xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'Output frequency (Hz)' }, gridcolor: '#edf1f3' }, yaxis: { title: { text: 'Level (dB re 1 RMS model unit)' }, gridcolor: '#edf1f3' }, shapes: markerItems.map(marker => ({ type: 'line', x0: marker.hz, x1: marker.hz, yref: 'paper', y0: 0, y1: 1, line: { color: marker.color, width: 1, dash: 'dot' } })), annotations: markerItems.map(marker => ({ x: Math.log10(marker.hz), yref: 'paper', y: 1.02, text: marker.name, showarrow: false, font: { size: 10, color: marker.color } })) }} />{result.markers && <div className="dp-readout">{result.conditions.map(condition => <div key={condition.id}><i className={`dot ${condition.id}`} /><span>{label(condition.id)} DP</span><strong>{condition.spectrum.dp_level_db?.toFixed(2) ?? '—'} <small>dB re 1</small></strong>{typeof condition.spectrum.dp_phase_rad === 'number' && <span className="phase-readout">Phase {condition.spectrum.dp_phase_rad.toFixed(2)} rad</span>}</div>)}</div>}<p className="panel-explanation">先生成完整時域 source 與逆向輸出，再分析頻譜。此路徑是現象學模型，未校準為耳道聲壓；兩條曲線使用相同分析窗與 Hann window；DP 數值以 least-squares 估計。Δ 表示修改模型減正常模型。</p></section>

    <section className="plot-panel"><PanelHeading number="04" title="單一 channel 的時間波形" english="20 ms channel inspection"><label className="compact-field"><span>CF</span><select aria-label="CF" value={selected} onChange={e => setSelected(Number(e.target.value))}>{result.cfs_hz.map((cf, i) => <option key={i} value={i}>{formatHz(cf)} · ch {i}</option>)}</select></label></PanelHeading><p className="excerpt-note">分析起點的 20 ms 原始取樣 · {(traceStart * 1000).toFixed(0)}–{((traceStart + .02) * 1000).toFixed(0)} ms · {result.sample_rate.toLocaleString()} samples/s</p><div className="waveform-grid">{(['bm', 'source'] as const).map(kind => <div key={kind}><h3>{kind === 'bm' ? 'BM · AC-coupled response' : `Source · ${component}`}</h3><Plot label={`${formatHz(result.cfs_hz[selected])} ${kind} 20 ms 波形`} height={230} data={waveformTraces(kind)} layout={{ xaxis: { title: { text: 'Time (ms)' }, range: [traceStart * 1000, (traceStart + .02) * 1000], gridcolor: '#edf1f3' }, yaxis: { title: { text: 'Model unit' }, gridcolor: '#edf1f3' }, margin: { l: 58, r: 18, t: 35, b: 45 } }} /></div>)}</div></section>

    <details className="input-details"><summary>輸入刺激（中耳濾波前）<span>Pre-middle-ear signal</span></summary><div className="waveform-grid"><Plot label="中耳濾波前刺激的 min/max 波形概覽" height={230} data={[{ type: 'scatter', mode: 'lines', x: result.input.time_s.map(t => t * 1000), y: result.input.waveform, line: { color: REFERENCE, width: 1 }, name: 'Input' }]} layout={{ xaxis: { title: { text: 'Time (ms)' } }, yaxis: { title: { text: 'Digital amplitude' } } }} /><Plot label="中耳濾波前刺激的頻譜 dBFS" height={230} data={[{ type: 'scatter', mode: 'lines', x: result.input.frequency_hz, y: result.input.spectrum_dbfs, line: { color: REFERENCE, width: 1 }, name: 'Input' }]} layout={{ xaxis: { type: 'log', ...FREQUENCY_TICKS, title: { text: 'Frequency (Hz)' } }, yaxis: { title: { text: 'Input spectrum (dBFS)' } } }} /></div><p className="panel-explanation">波形為保留 min/max 極值的降採樣概覽，不是完整 48 kHz 原始取樣。此處顯示中耳濾波前的輸入刺激；啟用中耳濾波時，它不等於 cochlear drive。</p></details>
    {result.warnings.length > 0 && <div className="warnings"><strong>本次運算備註</strong><ul>{result.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul></div>}
  </>;
}
