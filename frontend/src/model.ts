import type { AudioMetadata, FormSettings, RunRequest } from './types';

export const defaults: FormSettings = {
  stimulusKind: 'two_tone', f1: 1000, f2: 1220, l1: 65, l2: 55, toneDuration: .4,
  audioStart: 0, audioDuration: .4, audioLevel: 65, audioChannel: 'mix',
  health: { mode: 'global', health: .4, low_hz: 1000, high_hz: 4000 },
  analysisStart: .1, analysisEnd: .35, middleEar: 'none',
};

export function makeRequest(form: FormSettings, audio: AudioMetadata | null): RunRequest {
  const duration = form.stimulusKind === 'two_tone' ? form.toneDuration : form.audioDuration;
  if (!(duration >= .05 && duration <= 3)) throw new Error('片段長度需介於 0.05 與 3 秒。');
  if (!(form.analysisStart >= 0 && form.analysisEnd > form.analysisStart && form.analysisEnd <= duration)) {
    throw new Error('分析窗需位於片段內，且結束時間大於開始時間。');
  }
  const analysisSamples = Math.round(form.analysisEnd * 48000) - Math.round(form.analysisStart * 48000);
  if (analysisSamples < 960) {
    throw new Error('分析窗至少需要 20 ms（960 samples）。');
  }
  if (!(form.health.health >= 0 && form.health.health <= 1)) throw new Error('OHC health 需介於 0 與 1。');
  if (!(form.health.low_hz >= 0 && form.health.high_hz <= 24000)) throw new Error('CF 下界不可小於 0，且上界不可超過 24,000 Hz。');
  if (form.health.mode === 'band' && !(form.health.high_hz > form.health.low_hz)) {
    throw new Error('CF 頻帶上界需大於下界。');
  }
  let stimulus: RunRequest['stimulus'];
  if (form.stimulusKind === 'two_tone') {
    if (!(form.f1 >= 100 && form.f2 > form.f1 && form.f2 < 24000)) throw new Error('需符合 100 Hz ≤ f₁ < f₂ < 24,000 Hz。');
    const slowFrequency = Math.min(form.f2 - form.f1, 2 * form.f1 - form.f2);
    if (!(slowFrequency > 0)) throw new Error('請設定 f₂ < 2f₁，使 2f₁ − f₂ 為正頻率。');
    if (slowFrequency * analysisSamples / 48000 < 2) {
      throw new Error('分析窗至少需包含兩個 beat / DP 週期；請加長分析窗或調整雙音頻率。');
    }
    if (![form.l1, form.l2].every(v => Number.isFinite(v) && v >= 0 && v <= 100)) throw new Error('輸入 level 需介於 0 與 100 dB SPL。');
    stimulus = { kind: 'two_tone', f1_hz: form.f1, f2_hz: form.f2, l1_db_spl: form.l1, l2_db_spl: form.l2, duration_s: duration, ramp_ms: 20 };
  } else {
    if (!audio) throw new Error('請先選擇並上傳 WAV 或 FLAC 音訊。');
    if (!(form.audioStart >= 0 && form.audioStart + duration <= audio.duration_s + 1e-6)) throw new Error('選取片段超出音訊長度。');
    if (form.audioChannel === 'right' && audio.channels < 2) throw new Error('此音訊只有一個 channel，請選混音或左聲道。');
    if (!(form.audioLevel >= 0 && form.audioLevel <= 100)) throw new Error('音訊目標 level 需介於 0 與 100 dB SPL。');
    stimulus = { kind: 'audio', audio_id: audio.audio_id, start_s: form.audioStart, duration_s: duration, level_db_spl: form.audioLevel, channel: form.audioChannel };
  }
  return { stimulus, health: { ...form.health }, analysis: { start_s: form.analysisStart, end_s: form.analysisEnd }, middle_ear: form.middleEar };
}

export function commonBounds(matrices: number[][][]): [number, number] {
  const values = matrices.flat(2).filter(Number.isFinite);
  if (!values.length) return [-120, 0];
  const high = Math.ceil(values.reduce((max, value) => Math.max(max, value), -Infinity) / 5) * 5;
  if (high <= -300) return [-300, -280];
  // One common 80 dB range retains comparability and avoids a silence floor dominating the map.
  return [high - 80, high || 0];
}

export function formatHz(hz: number): string {
  return hz >= 1000 ? `${(hz / 1000).toFixed(hz % 1000 ? 2 : 0)} kHz` : `${Math.round(hz)} Hz`;
}

export function summarizeSettings(settings: Partial<RunRequest>, audio?: Partial<AudioMetadata>): string {
  const pieces: string[] = [];
  const stimulus = settings.stimulus;
  if (stimulus?.kind === 'two_tone') {
    pieces.push(`f₁/f₂ ${stimulus.f1_hz.toLocaleString()}/${stimulus.f2_hz.toLocaleString()} Hz`, `L₁/L₂ ${stimulus.l1_db_spl}/${stimulus.l2_db_spl} dB SPL`, `${stimulus.duration_s} s`);
  } else if (stimulus?.kind === 'audio') {
    const channels = { mix: 'Mix', left: 'Left', right: 'Right' };
    pieces.push(audio?.name ?? 'Uploaded audio', channels[stimulus.channel], `${stimulus.start_s.toFixed(2)}–${(stimulus.start_s + stimulus.duration_s).toFixed(2)} s`, `RMS ${stimulus.level_db_spl} dB SPL`);
  }
  if (settings.health) {
    const health = settings.health;
    pieces.push(`OHC ${health.mode === 'global' ? '全頻' : `${formatHz(health.low_hz)}–${formatHz(health.high_hz)}`} = ${health.health.toFixed(2)}`);
  }
  if (settings.middle_ear) pieces.push(`Middle ear: ${settings.middle_ear === 'none' ? 'None' : 'Verhulst 2018'}`);
  return pieces.length ? pieces.join(' · ') : '此結果未提供設定摘要，請下載結果 JSON 檢查。';
}

export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : typeof body.error === 'string' ? body.error : JSON.stringify(body.detail ?? body);
    } catch { /* Keep HTTP status for non-JSON failures. */ }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}
