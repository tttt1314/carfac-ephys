import { useEffect, useRef, useState } from 'react';

export type Trace = Record<string, unknown>;
type Props = { data: Trace[]; layout?: Record<string, unknown>; height?: number; label: string; onChannelClick?: (hz: number) => void };
type PlotEvent = { points: { y: number | string }[] };
type PlotElement = HTMLDivElement & { on?: (event: string, handler: (event: PlotEvent) => void) => void; removeAllListeners?: (event: string) => void };
const plotly = () => import('plotly.js-dist-min').then(module => module.default);

export default function Plot({ data, layout = {}, height = 280, label, onChannelClick }: Props) {
  const root = useRef<PlotElement>(null);
  const [error, setError] = useState('');
  const clickRef = useRef(onChannelClick);
  clickRef.current = onChannelClick;
  useEffect(() => {
    const element = root.current;
    if (!element) return;
    let cancelled = false;
    void plotly().then(async library => {
      if (cancelled) return;
      await library.react(element, data, {
        autosize: true, height,
        paper_bgcolor: 'transparent', plot_bgcolor: '#fff',
        font: { family: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif', size: 11, color: '#566878' },
        margin: { l: 60, r: 25, t: 16, b: 48 },
        xaxis: { gridcolor: '#edf1f3', zerolinecolor: '#dce4e7' },
        yaxis: { gridcolor: '#edf1f3', zerolinecolor: '#dce4e7' },
        legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 11 } },
        hovermode: 'closest', ...layout,
      }, { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'], toImageButtonOptions: { format: 'svg', filename: 'carfac-explorer' } });
      if (cancelled) return;
      element.removeAllListeners?.('plotly_click');
      element.on?.('plotly_click', event => {
        const hz = Number(event.points?.[0]?.y);
        if (Number.isFinite(hz)) clickRef.current?.(hz);
      });
    }).catch(reason => { if (!cancelled) setError(String(reason)); });
    return () => { cancelled = true; };
  }, [data, layout, height]);
  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const observer = new ResizeObserver(() => { void plotly().then(p => { if (element.isConnected) p.Plots.resize(element); }); });
    observer.observe(element);
    return () => { observer.disconnect(); void plotly().then(p => p.purge(element)); };
  }, []);
  return <div className="plot-wrap" role="img" aria-label={label} style={{ minHeight: height }}>
    <div ref={root} className="plot" />
    {error && <p className="plot-error">圖表載入失敗：{error}</p>}
  </div>;
}
