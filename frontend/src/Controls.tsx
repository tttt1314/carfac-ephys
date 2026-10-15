import type { AudioMetadata, FormSettings, Health } from './types';
import { formatHz } from './model';

type Props = { form: FormSettings; setForm: (form: FormSettings) => void; audio: AudioMetadata | null; uploading: boolean; upload: (file: File) => void; busy: boolean; run: () => void; tab: 'explore' | 'reference' };
type NumberFieldProps = { label: string; value: number; unit?: string; min?: number; max?: number; step?: number; onChange: (value: number) => void };

function NumberField({ label, value, unit, min, max, step = 1, onChange }: NumberFieldProps) {
  return <label className="field"><span>{label}</span><div className="input-unit"><input type="number" aria-label={label} value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} /><span>{unit}</span></div></label>;
}

export default function Controls({ form, setForm, audio, uploading, upload, busy, run, tab }: Props) {
  const update = <K extends keyof FormSettings>(key: K, value: FormSettings[K]) => setForm({ ...form, [key]: value });
  const updateHealth = <K extends keyof Health>(key: K, value: Health[K]) => update('health', { ...form.health, [key]: value });
  return <aside className="sidebar" aria-label="實驗設定">
    <div className="sidebar-title"><span className="eyebrow">EXPERIMENT SETUP</span><span className="local-dot">本機運算</span></div>
    <fieldset disabled={busy || uploading}>
      {tab === 'explore' && <section className="control-section">
        <div className="section-label"><span className="step-number">01</span><h2>輸入刺激 <small>Stimulus</small></h2></div>
        <div className="segmented" aria-label="刺激種類">
          <button type="button" aria-pressed={form.stimulusKind === 'two_tone'} onClick={() => update('stimulusKind', 'two_tone')}>雙音 Two-tone</button>
          <button type="button" aria-pressed={form.stimulusKind === 'audio'} onClick={() => update('stimulusKind', 'audio')}>上傳音訊</button>
        </div>
        {form.stimulusKind === 'two_tone' ? <>
          <div className="field-grid"><NumberField label="f₁" value={form.f1} unit="Hz" min={100} max={23999} onChange={v => update('f1', v)} /><NumberField label="f₂" value={form.f2} unit="Hz" min={100} max={23999} onChange={v => update('f2', v)} /></div>
          <div className="field-grid"><NumberField label="L₁" value={form.l1} unit="dB SPL" min={0} max={100} onChange={v => update('l1', v)} /><NumberField label="L₂" value={form.l2} unit="dB SPL" min={0} max={100} onChange={v => update('l2', v)} /></div>
          <NumberField label="刺激長度" value={form.toneDuration} unit="s" min={.05} max={3} step={.05} onChange={v => update('toneDuration', v)} />
          <div className="parameter-note"><span>f₂ / f₁ <strong>{form.f1 > 0 ? (form.f2 / form.f1).toFixed(3) : '—'}</strong></span><span>2f₁ − f₂ <strong>{formatHz(2 * form.f1 - form.f2)}</strong></span></div>
          <p className="field-help">起訖各 20 ms ramp。頻率標記只用於刺激與輸出分析，不傳給 source extraction。</p>
        </> : <>
          <label className={`upload-zone ${uploading ? 'uploading' : ''}`}>
            <span className="upload-icon" aria-hidden="true">↥</span><strong>{uploading ? '正在上傳…' : audio ? audio.name : '選擇音訊檔案'}</strong><span>WAV / FLAC · 只在本機處理</span>
            <input type="file" accept=".wav,.flac,audio/wav,audio/flac" aria-label="上傳 WAV 或 FLAC" onChange={e => { const file = e.target.files?.[0]; if (file) upload(file); e.target.value = ''; }} />
          </label>
          {audio && <p className="file-meta">{audio.duration_s.toFixed(2)} s · {(audio.sample_rate / 1000).toFixed(1)} kHz · {audio.channels} ch</p>}
          <div className="field-grid"><NumberField label="片段起點" value={form.audioStart} unit="s" min={0} step={.1} onChange={v => update('audioStart', v)} /><NumberField label="片段長度" value={form.audioDuration} unit="s" min={.05} max={3} step={.1} onChange={v => update('audioDuration', v)} /></div>
          <NumberField label="目標 RMS level" value={form.audioLevel} unit="dB SPL" min={0} max={100} onChange={v => update('audioLevel', v)} />
          <label className="field"><span>輸入聲道</span><select aria-label="輸入聲道" value={form.audioChannel} onChange={e => update('audioChannel', e.target.value as FormSettings['audioChannel'])}><option value="mix">混音 Mix</option><option value="left">左聲道 Left</option><option value="right" disabled={!audio || audio.channels < 2}>右聲道 Right</option></select></label>
          <p className="field-help">音訊會重取樣至 48 kHz 並縮放至指定 RMS level；這是模型輸入校準假設。</p>
        </>}
      </section>}

      {tab === 'reference' && <section className="control-section reference-setup"><div className="section-label"><span className="step-number">01</span><h2>參考刺激 <small>Reference</small></h2></div><p>使用參考資料的頻率與 primary levels，執行 24 點雙音比較。</p><p className="field-help">實測 FPL 與模型 nominal SPL 的定義不同。此比較用來檢查頻率趨勢，不能視為絕對聲壓校準。</p></section>}

      <section className="control-section">
        <div className="section-label"><span className="step-number">02</span><h2>OHC 比較 <small>Health</small></h2></div>
        <div className="condition-guide"><span><i className="dot normal" />正常模型 <strong>1.00</strong></span><span><i className="dot impaired" />修改模型 <strong>{form.health.health.toFixed(2)}</strong></span></div>
        <div className="segmented"><button type="button" aria-pressed={form.health.mode === 'global'} onClick={() => updateHealth('mode', 'global')}>全頻 Global</button><button type="button" aria-pressed={form.health.mode === 'band'} onClick={() => updateHealth('mode', 'band')}>CF 頻帶</button></div>
        {form.health.mode === 'band' && <div className="field-grid"><NumberField label="CF 下界" value={form.health.low_hz} unit="Hz" min={0} max={24000} step={100} onChange={v => updateHealth('low_hz', v)} /><NumberField label="CF 上界" value={form.health.high_hz} unit="Hz" min={0} max={24000} step={100} onChange={v => updateHealth('high_hz', v)} /></div>}
        <label className="range-field"><span>修改模型 OHC health <strong>{form.health.health.toFixed(2)}</strong></span><input type="range" aria-label="修改模型 OHC health" min={0} max={1} step={.01} value={form.health.health} onChange={e => updateHealth('health', Number(e.target.value))} /><span className="range-ticks"><span>0 · 被動</span><span>1 · 正常參數</span></span></label>
        <p className="field-help">Health 控制模型的主動增益，不代表存活細胞比例。頻帶外維持 1。</p>
      </section>

      <section className="control-section">
        <div className="section-label"><span className="step-number">03</span><h2>分析與傳播 <small>Analysis</small></h2></div>
        {tab === 'explore' && <div className="field-grid"><NumberField label="分析起點" value={form.analysisStart} unit="s" min={0} step={.01} onChange={v => update('analysisStart', v)} /><NumberField label="分析終點" value={form.analysisEnd} unit="s" min={0} step={.01} onChange={v => update('analysisEnd', v)} /></div>}
        <label className="field"><span>中耳濾波 Middle-ear filter</span><select aria-label="中耳濾波 Middle-ear filter" value={form.middleEar} onChange={e => update('middleEar', e.target.value as FormSettings['middleEar'])}><option value="none">無 None</option><option value="Verhulst_2018">Verhulst 2018</option></select></label>
        <p className="field-help">逆向傳播仍是研究用假設；模型輸出尚未校準為 Pa 或 dB SPL。</p>
        <details className="fixed-config"><summary>固定運算設定</summary><dl><div><dt>Sample rate</dt><dd>48,000 Hz</dd></div><div><dt>Precision</dt><dd>float64 / x64</dd></div><div><dt>AGC decimation</dt><dd>1 / 1 / 1 / 1</dd></div><div><dt>Forward delay</dt><dd>1 sample / stage</dd></div></dl><p>使用 full-rate AGC。首次執行包含 JAX compilation，可能需要較長時間。</p></details>
      </section>
    </fieldset>
    <div className="run-dock"><button className="run-button" type="button" onClick={run} disabled={busy || uploading} aria-label={tab === 'explore' ? '執行 CARFAC' : '執行參考比較'}>{busy ? <><span className="spinner" />計算中…</> : <><span aria-hidden="true">▶</span>{tab === 'explore' ? '執行 CARFAC' : '執行參考比較'}<kbd>Run</kbd></>}</button><p>設定修改後，按 Run 更新結果。</p></div>
  </aside>;
}
