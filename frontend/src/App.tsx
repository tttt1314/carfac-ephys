import { useEffect, useRef, useState } from 'react';
import Controls from './Controls';
import ExploreView, { EmptyExplore } from './ExploreView';
import ReferenceView from './ReferenceView';
import PathComparisonControls from './PathComparisonControls';
import PathComparisonView from './PathComparisonView';
import { makePathComparisonRequest } from './pathComparison';
import { api, defaults, makeRequest, summarizeSettings } from './model';
import type { AudioMetadata, BenchmarkResult, ExploreResult, FormSettings, Job, PathComparisonResult, Reference, RunRequest, WorkspaceTab } from './types';

export default function App() {
  const [tab, setTab] = useState<WorkspaceTab>(() => {
    const requested = new URLSearchParams(window.location.search).get('tab');
    return requested === 'reference' || requested === 'validation' ? requested : 'explore';
  });
  const [form, setForm] = useState<FormSettings>(defaults);
  const [audio, setAudio] = useState<AudioMetadata | null>(null);
  const [uploading, setUploading] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState<ExploreResult | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [pathResult, setPathResult] = useState<PathComparisonResult | null>(null);
  const [pathReference, setPathReference] = useState<Reference | null>(null);
  const [pathJobId, setPathJobId] = useState('');
  const [pathLoading, setPathLoading] = useState(false);
  const [pathError, setPathError] = useState('');
  const [pathAttempt, setPathAttempt] = useState(0);
  const [pathMiddleEar, setPathMiddleEar] = useState<FormSettings['middleEar']>('none');
  const [pathResultSignature, setPathResultSignature] = useState('');
  const pathInitialized = useRef(false);
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
  const pathSignature = JSON.stringify(makePathComparisonRequest(pathMiddleEar));
  const stale = tab === 'explore' ? !!result && signature !== resultSignature : tab === 'reference' ? !!benchmark && healthSignature !== benchmarkSignature : !!pathResult && pathSignature !== pathResultSignature;
  const runLabel = tab === 'explore' ? '執行 CARFAC' : tab === 'reference' ? '執行參考比較' : '執行四路徑比較';
  const displayedJobId = tab === 'explore' ? resultJobId : tab === 'reference' ? benchmarkJobId : pathJobId;

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', tab);
    window.history.replaceState(null, '', url);
  }, [tab]);

  useEffect(() => {
    if (tab !== 'validation') return;
    const controller = new AbortController();
    setPathLoading(true); setPathError('');
    void Promise.allSettled([
      api<Reference>('/api/path-comparisons/reference', { signal: controller.signal }),
      api<Job | null>('/api/path-comparisons/latest', { signal: controller.signal }),
    ]).then(([referenceResponse, latestResponse]) => {
      if (controller.signal.aborted) return;
      const errors: string[] = [];
      if (referenceResponse.status === 'fulfilled') setPathReference(referenceResponse.value);
      else errors.push(`參考資料：${String(referenceResponse.reason)}`);
      if (latestResponse.status === 'fulfilled') {
        const latest = latestResponse.value;
        if (latest?.status === 'complete' && latest.result?.kind === 'path_comparison') {
          setPathResult(latest.result); setPathJobId(latest.id);
          setPathResultSignature(JSON.stringify({ point_count: latest.result.settings.point_count, middle_ear: latest.result.settings.middle_ear }));
          if (!pathInitialized.current) setPathMiddleEar(latest.result.settings.middle_ear);
        }
      } else errors.push(`最近結果：${String(latestResponse.reason)}`);
      pathInitialized.current = true;
      setPathError(errors.join('；'));
      setPathLoading(false);
    });
    return () => controller.abort();
  }, [tab, pathAttempt]);

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
          else if (next.result.kind === 'benchmark') { setBenchmark(next.result); setBenchmarkSignature(requestSignature.current); setBenchmarkJobId(next.id); }
          else { setPathResult(next.result); setPathJobId(next.id); setPathResultSignature(requestSignature.current); }
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
      } else if (tab === 'reference') {
        if (!reference) throw new Error('參考資料尚未載入。');
        if (!(form.health.health >= 0 && form.health.health <= 1)) throw new Error('OHC health 需介於 0 與 1。');
        if (form.health.mode === 'band' && !(form.health.low_hz >= 0 && form.health.high_hz > form.health.low_hz && form.health.high_hz <= 24000)) throw new Error('請確認 CF 頻帶上下界。');
        body = { health: form.health, middle_ear: form.middleEar, point_count: 24 }; requestSignature.current = healthSignature;
      } else {
        if (!pathReference && !pathResult) throw new Error('路徑驗證參考資料尚未載入。');
        body = makePathComparisonRequest(pathMiddleEar); requestSignature.current = pathSignature;
      }
      setSubmitting(true);
      const endpoint = tab === 'explore' ? '/api/runs' : tab === 'reference' ? '/api/benchmarks' : '/api/path-comparisons';
      const created = await api<{ id: string }>(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      setJob({ id: created.id, status: 'queued', progress: 0, message: '已排入本機運算佇列。' });
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    finally { setSubmitting(false); }
  };

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="CARFAC Explorer 首頁"><svg viewBox="0 0 36 36" width="34" height="34" aria-hidden="true"><rect width="36" height="36" rx="10" fill="#087f8c" /><path d="M8 19h4l3-8 5 16 4-12 3 4h2" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg><div><strong>CARFAC <span>Explorer</span></strong><small>耳蝸訊號工作台</small></div></a><nav className="main-tabs" aria-label="工作區"><button className={tab === 'explore' ? 'active' : ''} onClick={() => setTab('explore')} aria-current={tab === 'explore' ? 'page' : undefined}>探索 <span>Explore</span></button><button className={tab === 'reference' ? 'active' : ''} onClick={() => setTab('reference')} aria-current={tab === 'reference' ? 'page' : undefined}>實測參考 <span>Reference</span></button><button className={tab === 'validation' ? 'active' : ''} onClick={() => setTab('validation')} aria-current={tab === 'validation' ? 'page' : undefined}>路徑驗證 <span>Validation</span></button></nav><div className="topbar-meta"><span className="research-tag">RESEARCH PROTOTYPE</span><span className="version">v0.1</span></div></header>
    <div className="workspace">{tab === 'validation' ? <PathComparisonControls middleEar={pathMiddleEar} setMiddleEar={setPathMiddleEar} busy={busy || pathLoading} run={() => void run()} hasResult={!!pathResult} /> : <Controls form={form} setForm={setForm} audio={audio} uploading={uploading} upload={file => void upload(file)} busy={busy} run={() => void run()} tab={tab} />}<main className="main-content">
      <div className="workspace-heading"><div><span className="eyebrow">{tab === 'explore' ? 'SIGNAL EXPLORATION' : tab === 'reference' ? 'EMPIRICAL REFERENCE' : 'REVERSE-PATH VALIDATION'}</span><h1>{tab === 'explore' ? '從刺激到非線性輸出' : tab === 'reference' ? '模型與實測的對照' : '四條路徑，保留頻帶驗證'}</h1><p>{tab === 'explore' ? '同一刺激、兩種 OHC 設定。沿著訊號路徑比較反應。' : tab === 'reference' ? '先保留資料的單位與限制，再檢查模型的頻率趨勢。' : '固定同一份 sources，檢查延遲與 CF 濾波是否改善未參與校準的頻帶。'}</p></div><div className="engine-tags"><span>48 kHz</span><span>float64</span><span>AGC full-rate</span></div></div>
      {error && <div className="error-box" role="alert"><strong>無法完成此操作</strong><p>{error}</p><button className="dismiss-error" onClick={() => setError('')} aria-label="關閉錯誤">×</button></div>}
      {busy && <div className="progress-card" role="status" aria-live="polite"><div><span className="spinner" /><strong>{submitting ? '送出運算設定…' : job?.message || 'CARFAC 計算中…'}</strong><span>{Math.round(Math.max(0, Math.min(1, job?.progress ?? 0)) * 100)}%</span></div><progress max={1} value={job?.progress ?? 0} /><p>運算在本機執行。第一次 JAX 編譯可能比後續執行久；請保留此頁面。</p></div>}
      {stale && <div className="stale-notice" role="status"><span aria-hidden="true">↻</span><div><strong>設定已修改，以下仍是上一次的結果。</strong><span>按「{runLabel}」套用目前設定。</span></div></div>}
      {tab === 'explore' && result && <div className="run-context"><span className="status-dot" /><strong>已完成設定</strong><p>{summarizeSettings(result.settings as Partial<RunRequest>, result.input_audio)}</p></div>}
      {tab === 'reference' && benchmark?.settings && <div className="run-context"><span className="status-dot" /><strong>已完成設定</strong><p>{summarizeSettings(benchmark.settings as Partial<RunRequest>)}</p></div>}
      {tab === 'validation' && pathResult && <div className="run-context"><span className="status-dot" /><strong>已完成設定</strong><p>{pathResult.dp_hz.length} points · OHC health = 1 · 無 reflection · Forward middle ear: {pathResult.settings.middle_ear === 'none' ? 'None' : 'Verhulst 2018'} · Reverse middle ear: fixed loaded-stapes</p></div>}
      {displayedJobId && <div className="result-actions"><span>下載目前顯示結果，包含該次執行的設定與數值。</span><a className="export-link" href={`/api/runs/${displayedJobId}/export`} download={`carfac-${tab}-${displayedJobId}.json`}>↓ 下載結果 JSON</a></div>}
      {tab === 'explore' ? result ? <ExploreView result={result} /> : <EmptyExplore /> : tab === 'reference' ? <ReferenceView reference={reference} loading={referenceLoading} error={referenceError} retry={() => setReferenceAttempt(n => n + 1)} benchmark={benchmark} /> : <PathComparisonView reference={pathReference} result={pathResult} loading={pathLoading} error={pathError} retry={() => setPathAttempt(n => n + 1)} />}
      <footer className="workspace-footer"><span>CARFAC / 本機研究工具</span><span>BM、source 與 emission 都有不同的訊號定義；比較前請確認單位。</span></footer>
    </main></div>
  </div>;
}
