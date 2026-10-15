import { useEffect, useRef, useState } from 'react';
import Controls from './Controls';
import ExploreView, { EmptyExplore } from './ExploreView';
import ReferenceView from './ReferenceView';
import { api, defaults, makeRequest, summarizeSettings } from './model';
import type { AudioMetadata, BenchmarkResult, ExploreResult, FormSettings, Job, Reference, RunRequest } from './types';

export default function App() {
  const [tab, setTab] = useState<'explore' | 'reference'>('explore');
  const [form, setForm] = useState<FormSettings>(defaults);
  const [audio, setAudio] = useState<AudioMetadata | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ExploreResult | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [resultJobId, setResultJobId] = useState('');
  const [benchmarkJobId, setBenchmarkJobId] = useState('');
  const [reference, setReference] = useState<Reference | null>(null);
  const [referenceLoading, setReferenceLoading] = useState(false);
  const [referenceError, setReferenceError] = useState('');
  const [referenceAttempt, setReferenceAttempt] = useState(0);
  const [resultSignature, setResultSignature] = useState('');
  const [benchmarkSignature, setBenchmarkSignature] = useState('');
  const requestSignature = useRef('');
  const [submitting, setSubmitting] = useState(false);
  const busy = submitting || job?.status === 'queued' || job?.status === 'running';
  const signature = JSON.stringify({ form, audio_id: audio?.audio_id });
  const healthSignature = JSON.stringify({ health: form.health, middle_ear: form.middleEar });
  const stale = tab === 'explore' ? !!result && signature !== resultSignature : !!benchmark && healthSignature !== benchmarkSignature;

  useEffect(() => {
    if (tab !== 'reference' || reference) return;
    const controller = new AbortController();
    setReferenceLoading(true); setReferenceError('');
    void api<Reference>('/api/reference', { signal: controller.signal }).then(setReference).catch(reason => {
      if (!controller.signal.aborted) setReferenceError(reason instanceof Error ? reason.message : String(reason));
    }).finally(() => { if (!controller.signal.aborted) setReferenceLoading(false); });
    return () => controller.abort();
  }, [tab, reference, referenceAttempt]);

  useEffect(() => {
    if (!job?.id || job.status === 'complete' || job.status === 'error') return;
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await api<Job>(`/api/runs/${job.id}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setJob(next);
        if (next.status === 'complete') {
          if (!next.result) { setError('運算已結束，但伺服器沒有回傳結果。'); return; }
          if (next.result.kind === 'explore') { setResult(next.result); setResultSignature(requestSignature.current); setResultJobId(next.id); }
          else { setBenchmark(next.result); setBenchmarkSignature(requestSignature.current); setBenchmarkJobId(next.id); }
        } else if (next.status === 'error') setError(next.error ?? next.message ?? '運算失敗。');
        else timeout = setTimeout(poll, 800);
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(`無法取得運算狀態：${reason instanceof Error ? reason.message : String(reason)}`);
        setJob(previous => previous ? { ...previous, status: 'error' } : null);
      }
    };
    timeout = setTimeout(poll, 200);
    return () => { controller.abort(); clearTimeout(timeout); };
    // One polling lifecycle per job; local status updates do not restart the timer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id]);

  const upload = async (file: File) => {
    if (!/\.(wav|flac)$/i.test(file.name)) { setError('請選擇 WAV 或 FLAC 檔案。'); return; }
    setUploading(true); setError('');
    const body = new FormData(); body.append('file', file);
    try {
      const metadata = await api<AudioMetadata>('/api/audio', { method: 'POST', body });
      setAudio(metadata);
      const duration = Math.floor(Math.min(metadata.duration_s, 3) * 1000) / 1000;
      setForm(previous => ({ ...previous, stimulusKind: 'audio', audioStart: 0, audioDuration: duration, audioChannel: 'mix', analysisStart: Math.min(.1, duration / 4), analysisEnd: Math.min(duration, Math.max(duration / 2, duration - .02)) }));
    } catch (reason) { setError(`音訊上傳失敗：${reason instanceof Error ? reason.message : String(reason)}`); }
    finally { setUploading(false); }
  };

  const run = async () => {
    setError('');
    try {
      let body: object;
      if (tab === 'explore') {
        const request = makeRequest(form, audio);
        body = request; requestSignature.current = signature;
      } else {
        if (!reference) throw new Error('參考資料尚未載入。');
        if (!(form.health.health >= 0 && form.health.health <= 1)) throw new Error('OHC health 需介於 0 與 1。');
        if (form.health.mode === 'band' && !(form.health.low_hz >= 0 && form.health.high_hz > form.health.low_hz && form.health.high_hz <= 24000)) throw new Error('請確認 CF 頻帶上下界。');
        body = { health: form.health, middle_ear: form.middleEar, point_count: 24 }; requestSignature.current = healthSignature;
      }
      setSubmitting(true);
      const created = await api<{ id: string }>(tab === 'explore' ? '/api/runs' : '/api/benchmarks', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      setJob({ id: created.id, status: 'queued', progress: 0, message: '已排入本機運算佇列。' });
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setSubmitting(false); }
  };

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="CARFAC Explorer 首頁"><svg viewBox="0 0 36 36" width="34" height="34" aria-hidden="true"><rect width="36" height="36" rx="10" fill="#087f8c" /><path d="M8 19h4l3-8 5 16 4-12 3 4h2" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg><div><strong>CARFAC <span>Explorer</span></strong><small>耳蝸訊號工作台</small></div></a><nav className="main-tabs" aria-label="工作區"><button className={tab === 'explore' ? 'active' : ''} onClick={() => setTab('explore')} aria-current={tab === 'explore' ? 'page' : undefined}>探索 <span>Explore</span></button><button className={tab === 'reference' ? 'active' : ''} onClick={() => setTab('reference')} aria-current={tab === 'reference' ? 'page' : undefined}>實測參考 <span>Reference</span></button></nav><div className="topbar-meta"><span className="research-tag">RESEARCH PROTOTYPE</span><span className="version">v0.1</span></div></header>
    <div className="workspace"><Controls form={form} setForm={setForm} audio={audio} uploading={uploading} upload={file => void upload(file)} busy={busy} run={() => void run()} tab={tab} /><main className="main-content">
      <div className="workspace-heading"><div><span className="eyebrow">{tab === 'explore' ? 'SIGNAL EXPLORATION' : 'EMPIRICAL REFERENCE'}</span><h1>{tab === 'explore' ? '從刺激到非線性輸出' : '模型與實測的對照'}</h1><p>{tab === 'explore' ? '同一刺激、兩種 OHC 設定。沿著訊號路徑比較反應。' : '先保留資料的單位與限制，再檢查模型的頻率趨勢。'}</p></div><div className="engine-tags"><span>48 kHz</span><span>float64</span><span>AGC full-rate</span></div></div>
      {error && <div className="error-box" role="alert"><strong>無法完成此操作</strong><p>{error}</p><button className="dismiss-error" onClick={() => setError('')} aria-label="關閉錯誤">×</button></div>}
      {busy && <div className="progress-card" role="status" aria-live="polite"><div><span className="spinner" /><strong>{submitting ? '送出運算設定…' : job?.message || 'CARFAC 計算中…'}</strong><span>{Math.round(Math.max(0, Math.min(1, job?.progress ?? 0)) * 100)}%</span></div><progress max={1} value={job?.progress ?? 0} /><p>運算在本機執行。第一次 JAX 編譯可能比後續執行久；請保留此頁面。</p></div>}
      {stale && <div className="stale-notice" role="status"><span aria-hidden="true">↻</span><div><strong>設定已修改，以下仍是上一次的結果。</strong><span>按「{tab === 'explore' ? '執行 CARFAC' : '執行參考比較'}」套用目前設定。</span></div></div>}
      {tab === 'explore' && result && <div className="run-context"><span className="status-dot" /><strong>已完成設定</strong><p>{summarizeSettings(result.settings as Partial<RunRequest>, result.input_audio)}</p></div>}
      {tab === 'reference' && benchmark?.settings && <div className="run-context"><span className="status-dot" /><strong>已完成設定</strong><p>{summarizeSettings(benchmark.settings as Partial<RunRequest>)}</p></div>}
      {(tab === 'explore' ? result && resultJobId : benchmark && benchmarkJobId) && <div className="result-actions"><span>下載目前顯示結果，包含該次執行的設定與數值。</span><a className="export-link" href={`/api/runs/${tab === 'explore' ? resultJobId : benchmarkJobId}/export`} download={`carfac-${tab}-${tab === 'explore' ? resultJobId : benchmarkJobId}.json`}>↓ 下載結果 JSON</a></div>}
      {tab === 'explore' ? result ? <ExploreView result={result} /> : <EmptyExplore /> : <ReferenceView reference={reference} loading={referenceLoading} error={referenceError} retry={() => setReferenceAttempt(n => n + 1)} benchmark={benchmark} />}
      <footer className="workspace-footer"><span>CARFAC / 本機研究工具</span><span>BM、source 與 emission 都有不同的訊號定義；比較前請確認單位。</span></footer>
    </main></div>
  </div>;
}
