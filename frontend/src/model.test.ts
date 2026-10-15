import { describe, expect, it } from 'vitest';
import { commonBounds, defaults, makeRequest, summarizeSettings } from './model';
import type { AudioMetadata } from './types';

describe('explorer requests', () => {
  it('sends complete physical settings only after explicit request construction', () => {
    const request = makeRequest(defaults, null);
    expect(request).toEqual({ stimulus: { kind: 'two_tone', f1_hz: 1000, f2_hz: 1220, l1_db_spl: 65, l2_db_spl: 55, duration_s: .4, ramp_ms: 20 }, health: { mode: 'global', health: .4, low_hz: 1000, high_hz: 4000 }, analysis: { start_s: .1, end_s: .35 }, middle_ear: 'none' });
    expect(request.stimulus).not.toHaveProperty('dp_hz');
  });
  it('preserves CF-band health and uploaded-audio segment settings', () => {
    const audio: AudioMetadata = { audio_id: 'local-123', name: 'test.flac', duration_s: 7, sample_rate: 44100, channels: 2 };
    const health = { mode: 'band' as const, health: .25, low_hz: 1800, high_hz: 6000 };
    const request = makeRequest({ ...defaults, stimulusKind: 'audio', audioStart: 2, audioDuration: .5, audioChannel: 'right', audioLevel: 60, health, middleEar: 'Verhulst_2018' }, audio);
    expect(request.stimulus).toEqual({ kind: 'audio', audio_id: 'local-123', start_s: 2, duration_s: .5, level_db_spl: 60, channel: 'right' });
    expect(request.health).toEqual(health);
    expect(request.middle_ear).toBe('Verhulst_2018');
    expect(request).not.toHaveProperty('markers');
  });
  it('rejects invalid analysis windows, CF bands and missing audio before a run', () => {
    expect(() => makeRequest({ ...defaults, analysisEnd: .6 }, null)).toThrow('分析窗');
    expect(() => makeRequest({ ...defaults, analysisEnd: .11 }, null)).toThrow('20 ms');
    expect(() => makeRequest({ ...defaults, health: { ...defaults.health, mode: 'band', high_hz: 500 } }, null)).toThrow('CF');
    expect(() => makeRequest({ ...defaults, stimulusKind: 'audio' }, null)).toThrow('上傳');
    expect(() => makeRequest({ ...defaults, f2: 900 }, null)).toThrow('f₁');
    expect(() => makeRequest({ ...defaults, f2: 2200 }, null)).toThrow('正頻率');
    expect(() => makeRequest({ ...defaults, f2: 1001 }, null)).toThrow('週期');
    expect(() => makeRequest({ ...defaults, toneDuration: 4 }, null)).toThrow('3 秒');
  });
  it('summarizes recorded settings without reading modified controls', () => {
    const completed = makeRequest(defaults, null);
    const changed = { ...defaults, f1: 1500, health: { ...defaults.health, health: .1 } };
    expect(summarizeSettings(completed)).toContain('1,000/1,220 Hz');
    expect(summarizeSettings(completed)).toContain('0.40');
    expect(summarizeSettings(completed)).not.toContain(changed.f1.toLocaleString());
    expect(() => makeRequest({ ...defaults, toneDuration: .05, analysisStart: .01, analysisEnd: .04 }, null)).not.toThrow();
  });
  it('keeps source heatmap scale shared across both conditions', () => {
    expect(commonBounds([[[-110, -40]], [[-100, -20]]])).toEqual([-100, -20]);
    expect(commonBounds([[[NaN]], [[-Infinity]]])).toEqual([-120, 0]);
    expect(commonBounds([[[-300]], [[-300]]])).toEqual([-300, -280]);
  });
});
