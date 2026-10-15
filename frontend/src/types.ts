export type Health = { mode: 'global' | 'band'; health: number; low_hz: number; high_hz: number };
export type AudioMetadata = { audio_id: string; name: string; duration_s: number; sample_rate: number; channels: number };
export type FormSettings = {
  stimulusKind: 'two_tone' | 'audio';
  f1: number; f2: number; l1: number; l2: number; toneDuration: number;
  audioStart: number; audioDuration: number; audioLevel: number; audioChannel: 'mix' | 'left' | 'right';
  health: Health; analysisStart: number; analysisEnd: number; middleEar: 'none' | 'Verhulst_2018';
};
export type RunRequest = {
  stimulus: { kind: 'two_tone'; f1_hz: number; f2_hz: number; l1_db_spl: number; l2_db_spl: number; duration_s: number; ramp_ms: number }
    | { kind: 'audio'; audio_id: string; start_s: number; duration_s: number; level_db_spl: number; channel: 'mix' | 'left' | 'right' };
  health: Health; analysis: { start_s: number; end_s: number }; middle_ear: FormSettings['middleEar'];
};
export type Spectrum = { frequency_hz: number[]; level_db: number[]; dp_level_db?: number; dp_phase_rad?: number };
export type Condition = {
  id: 'normal' | 'impaired'; label: string; health: number[];
  bm: { time_s: number[]; rms_db: number[][] };
  sources: { time_s: number[]; total_db: number[][]; fast_db: number[][]; agc_db: number[][]; gain_db: number[][] };
  spectrum: Spectrum;
  waveforms: { time_s: number[]; bm: number[][]; source_total: number[][]; source_fast: number[][]; source_agc: number[][]; source_gain: number[][] };
};
export type ExploreResult = {
  kind: 'explore'; sample_rate: number; duration_s: number; cfs_hz: number[];
  analysis_window_s: [number, number]; conditions: Condition[];
  input: { time_s: number[]; waveform: number[]; frequency_hz: number[]; spectrum_dbfs: number[] };
  markers?: { f1_hz: number; f2_hz: number; dp_hz: number };
  warnings: string[]; compute_seconds: number; settings: object;
  input_audio?: AudioMetadata;
};
export type Reference = {
  id: string; title: string; source_url: string; doi: string; license: string | { name?: string; label?: string; url?: string };
  component: string; stimulus: { ratio: number; l1_db_fpl: number; l2_db_fpl: number };
  points: { dp_hz: number; f1_hz: number; f2_hz: number; level_db_spl: number; phase_cycles: number }[];
  limitations: string[];
};
export type BenchmarkResult = {
  kind: 'benchmark'; reference: Reference; dp_hz: number[]; normal_db: number[]; impaired_db: number[];
  normal_phase_cycles: (number | null)[]; impaired_phase_cycles: (number | null)[]; reference_relative_db: number[];
  normal_relative_db: number[]; impaired_relative_db: number[]; warnings: string[]; compute_seconds: number;
  settings?: object;
};
export type Job = { id: string; status: 'queued' | 'running' | 'complete' | 'error'; progress: number; message: string; error?: string; result?: ExploreResult | BenchmarkResult };
export type SourceComponent = 'total' | 'fast' | 'agc' | 'gain';
