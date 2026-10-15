"""Reproducible stimuli and diagnostics for the local CARFAC/OAE explorer.

Explorer settings are deliberately independent of the core model defaults:
48 kHz, float64, audio-rate AGC, and fresh silent state for each condition.
Output levels use model units; the reverse path has no pressure calibration.
"""

import json
import time
from collections.abc import Callable
from functools import lru_cache
from importlib import metadata
from typing import Annotated, Any, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carfac_ephys import constants
from carfac_ephys.middle_ear import FILTERS, apply_middle_ear
from carfac_ephys.oae import DifferentiableOae, OaeDiagnostics
from carfac_ephys.oae_reference import load_reference

SAMPLE_RATE = 48000


def _model_metadata() -> dict[str, Any]:
  """Record fixed model choices and the installed upstream source revision."""
  direct_url = metadata.distribution("carfac").read_text("direct_url.json")
  source = json.loads(direct_url) if direct_url else {}
  return {
    "sample_rate": SAMPLE_RATE,
    "dtype": "float64",
    "agc_decimation": [1, 1, 1, 1],
    "velocity_scale": 0.1,
    "v_offset": 0.04,
    "source_coupling": 0.1,
    "reverse_q": 2**-0.5,
    "reverse_middle_ear_gain": 1.0,
    "jax_version": jax.__version__,
    "carfac_version": metadata.version("carfac"),
    "carfac_commit": source.get("vcs_info", {}).get("commit_id"),
    "output_units": "dB re 1 model unit (RMS)",
    "state": "fresh silent initialization per condition",
    "source_definition": "Full time-domain remainder relative to each stage's own resting linearization; includes fundamental compression.",
    "reverse_path": "Phenomenological one-way source propagation; no feedback into forward CARFAC.",
  }


class Settings(BaseModel):
  """Reject misspelled settings and non-finite numbers at the API boundary."""

  model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TwoTone(Settings):
  """Two independent RMS-calibrated tones with raised-cosine ramps."""

  kind: Literal["two_tone"] = "two_tone"
  f1_hz: float = Field(default=1000, ge=100, lt=SAMPLE_RATE / 2)
  f2_hz: float = Field(default=1220, ge=100, lt=SAMPLE_RATE / 2)
  l1_db_spl: float = Field(default=65, ge=0, le=100)
  l2_db_spl: float = Field(default=55, ge=0, le=100)
  duration_s: float = Field(default=0.4, ge=0.05, le=3)
  ramp_ms: float = Field(default=20, ge=0, le=100)

  @model_validator(mode="after")
  def check_tones(self) -> Self:
    """Keep the primaries ordered and the cubic difference frequency positive."""
    if not self.f1_hz < self.f2_hz < 2 * self.f1_hz:
      raise ValueError("Two-tone stimulus requires f1 < f2 < 2*f1.")
    if 2 * self.ramp_ms / 1000 > self.duration_s:
      raise ValueError("Onset and offset ramps exceed stimulus duration.")
    return self


class AudioStimulus(Settings):
  """A bounded excerpt from an uploaded recording, normalized to nominal SPL."""

  kind: Literal["audio"] = "audio"
  audio_id: str
  start_s: float = Field(default=0, ge=0)
  duration_s: float = Field(default=0.4, ge=0.05, le=3)
  level_db_spl: float = Field(default=65, ge=0, le=100)
  channel: Literal["mix", "left", "right"] = "mix"


class Health(Settings):
  """OHC model health globally or within a characteristic-frequency band."""

  mode: Literal["global", "band"] = "global"
  health: float = Field(default=0.4, ge=0, le=1)
  low_hz: float = Field(default=1000, ge=0)
  high_hz: float = Field(default=4000, le=SAMPLE_RATE / 2)

  @model_validator(mode="after")
  def check_band(self) -> Self:
    """Reject an empty or reversed frequency range."""
    if self.mode == "band" and self.low_hz >= self.high_hz:
      raise ValueError("OHC frequency band requires low_hz < high_hz.")
    return self


class Analysis(Settings):
  """Time window used for the output spectrum and DP measurement."""

  start_s: float = Field(default=0.1, ge=0)
  end_s: float = Field(default=0.35, gt=0)


class RunRequest(Settings):
  """Complete settings for a matched normal/impaired comparison."""

  stimulus: Annotated[TwoTone | AudioStimulus, Field(discriminator="kind")] = Field(
    default_factory=TwoTone
  )
  health: Health = Field(default_factory=Health)
  analysis: Analysis = Field(default_factory=Analysis)
  middle_ear: Literal["none", "Verhulst_2018"] = "none"

  @model_validator(mode="after")
  def check_window(self) -> Self:
    """Require a usable spectrum window wholly inside the selected stimulus."""
    if not 0 <= self.analysis.start_s < self.analysis.end_s <= self.stimulus.duration_s:
      raise ValueError("Analysis requires 0 <= start < end <= stimulus duration.")
    samples = round(self.analysis.end_s * SAMPLE_RATE) - round(self.analysis.start_s * SAMPLE_RATE)
    if samples < SAMPLE_RATE // 50:
      raise ValueError("Analysis window must be at least 20 ms.")
    if isinstance(self.stimulus, TwoTone):
      gap = min(
        self.stimulus.f2_hz - self.stimulus.f1_hz, 2 * self.stimulus.f1_hz - self.stimulus.f2_hz
      )
      if gap * samples / SAMPLE_RATE < 2:
        raise ValueError(
          "Analysis cannot resolve these tones: increase the window or frequency separation."
        )
    return self


def prepare_stimulus(request: RunRequest) -> np.ndarray:
  """Generate a two-tone waveform with per-tone RMS levels in repository units."""
  stimulus = request.stimulus
  if not isinstance(stimulus, TwoTone):
    raise ValueError("Audio stimulus requires a decoded recording.")
  count = round(stimulus.duration_s * SAMPLE_RATE)
  time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
  wave = np.zeros(count, dtype=np.float64)
  for hz, level in ((stimulus.f1_hz, stimulus.l1_db_spl), (stimulus.f2_hz, stimulus.l2_db_spl)):
    wave += np.sqrt(2) * constants.db_spl_to_amplitude(level) * np.sin(2 * np.pi * hz * time)
  ramp_count = round(stimulus.ramp_ms / 1000 * SAMPLE_RATE)
  if ramp_count:
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp_count))
    wave[:ramp_count] *= ramp
    wave[-ramp_count:] *= ramp[::-1]
  return np.asarray(wave, dtype=np.float64)


def health_profile(settings: Health, cfs: np.ndarray) -> np.ndarray:
  """Map the requested OHC condition to the actual descending CF axis."""
  values = np.ones_like(cfs, dtype=np.float64)
  selected = (
    np.ones(cfs.shape, dtype=bool)
    if settings.mode == "global"
    else (cfs >= settings.low_hz) & (cfs <= settings.high_hz)
  )
  if not selected.any():
    raise ValueError("OHC band contains no CARFAC channels; widen the frequency range.")
  values[selected] = settings.health
  return values


def spectrum(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Return a Hann-windowed one-sided RMS amplitude spectrum, not a PSD."""
  window = np.hanning(len(waveform))
  amplitude = np.abs(np.fft.rfft(waveform * window)) / window.sum()
  amplitude[1:] *= np.sqrt(2)
  if len(waveform) % 2 == 0:
    amplitude[-1] /= np.sqrt(2)
  levels = 20 * np.log10(np.maximum(amplitude, 1e-15))
  return np.fft.rfftfreq(len(waveform), 1 / SAMPLE_RATE), levels


def measure_dp(waveform: np.ndarray, start_sample: int, tones: TwoTone) -> complex:
  """Measure DP only at the final output, jointly fitting the two primaries.

  Weighted sine/cosine least squares reduces primary leakage when stimulus
  frequencies do not fall on FFT bins. Source extraction never sees these
  frequencies. The returned phasor is RMS with a cosine phase reference.
  """
  time_s = (start_sample + np.arange(len(waveform))) / SAMPLE_RATE
  frequencies = [tones.f1_hz, tones.f2_hz, 2 * tones.f1_hz - tones.f2_hz]
  columns = [np.ones(len(waveform))]
  for frequency in frequencies:
    phase = 2 * np.pi * frequency * time_s
    columns.extend([np.cos(phase), np.sin(phase)])
  design = np.column_stack(columns)
  weight = np.sqrt(np.hanning(len(waveform)))
  coefficients, _, rank, singular = np.linalg.lstsq(
    design * weight[:, None], waveform * weight, rcond=None
  )
  if rank < design.shape[1] or singular[0] / singular[-1] > 1e8:
    raise ValueError("DP fit is not identifiable: increase the window or frequency separation.")
  return complex(coefficients[-2], -coefficients[-1]) / np.sqrt(2)


@lru_cache(maxsize=1)
def _engine():
  """Cache compiled operations; each call still initializes its own silent state."""
  with jax.enable_x64():
    model = DifferentiableOae(fs=SAMPLE_RATE, agc_decimation=(1, 1, 1, 1))
  diagnostics = jax.jit(model.simulate_diagnostics)
  emission = jax.jit(lambda wave, parameters: model.simulate(wave, parameters).ear_canal)
  return model, diagnostics, emission


def _heatmap(values: np.ndarray) -> dict[str, Any]:
  """Compress time by 5-ms RMS integration while retaining every CF channel."""
  width = SAMPLE_RATE // 200
  starts = np.arange(0, len(values), width)
  counts = np.minimum(width, len(values) - starts)
  rms = np.sqrt(np.add.reduceat(values**2, starts, axis=0) / counts[:, None])
  return {
    "time_s": ((starts + (counts - 1) / 2) / SAMPLE_RATE).tolist(),
    "rms_db": np.round(20 * np.log10(np.maximum(rms, 1e-15)), 4).T.tolist(),
  }


def _condition_result(
  identifier: str,
  result: OaeDiagnostics,
  health: np.ndarray,
  request: RunRequest,
) -> dict[str, Any]:
  """Convert one condition to bounded plot data without retaining large state arrays."""
  bm = np.asarray(result.bm)
  components = {
    "total": np.asarray(result.sources.nonlinear),
    "fast": np.asarray(result.sources.fast),
    "agc": np.asarray(result.sources.agc),
    "gain": np.asarray(result.sources.gain),
  }
  output = np.asarray(result.ear_canal)
  if not np.isfinite(bm).all() or not np.isfinite(output).all():
    raise ValueError("The model produced non-finite values; reduce the stimulus level.")
  start, end = (
    round(value * SAMPLE_RATE) for value in (request.analysis.start_s, request.analysis.end_s)
  )
  stop = min(end, start + SAMPLE_RATE // 50)
  frequencies, levels = spectrum(output[start:end])
  output_spectrum: dict[str, Any] = {
    "frequency_hz": frequencies.tolist(),
    "level_db": levels.tolist(),
  }
  if isinstance(request.stimulus, TwoTone):
    dp = measure_dp(output[start:end], start, request.stimulus)
    output_spectrum["dp_level_db"] = float(20 * np.log10(max(abs(dp), 1e-15)))
    if abs(dp) > 1e-15:
      output_spectrum["dp_phase_rad"] = float(np.angle(dp))
  sources: dict[str, Any] = {}
  waveforms: dict[str, Any] = {
    "time_s": (np.arange(start, stop) / SAMPLE_RATE).tolist(),
    "bm": bm[start:stop].T.tolist(),
  }
  for name, values in components.items():
    if not np.isfinite(values).all():
      raise ValueError("The model produced non-finite source values.")
    data = _heatmap(values)
    sources["time_s"] = data["time_s"]
    sources[f"{name}_db"] = data["rms_db"]
    waveforms[f"source_{name}"] = values[start:stop].T.tolist()
  return {
    "id": identifier,
    "label": "Normal" if identifier == "normal" else "OHC impairment",
    "health": health.tolist(),
    "bm": _heatmap(bm),
    "sources": sources,
    "spectrum": output_spectrum,
    "waveforms": waveforms,
  }


def _input_preview(waveform: np.ndarray, request: RunRequest) -> dict[str, Any]:
  """Keep extrema for the overview and the complete analysis-window spectrum."""
  width = max(1, int(np.ceil(len(waveform) / 1000)))
  indices = []
  for start in range(0, len(waveform), width):
    part = waveform[start : start + width]
    indices.extend(sorted({start + int(np.argmin(part)), start + int(np.argmax(part))}))
  start = round(request.analysis.start_s * SAMPLE_RATE)
  end = round(request.analysis.end_s * SAMPLE_RATE)
  frequencies, levels = spectrum(waveform[start:end])
  return {
    "time_s": (np.asarray(indices) / SAMPLE_RATE).tolist(),
    "waveform": waveform[indices].tolist(),
    "frequency_hz": frequencies.tolist(),
    "spectrum_dbfs": levels.tolist(),
  }


def run_comparison(
  request: RunRequest,
  waveform: np.ndarray | None = None,
  progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
  """Run two independent, matched conditions and return scientific plot data."""
  started = time.monotonic()
  if waveform is None:
    waveform = prepare_stimulus(request)
  warnings = [
    "Reverse output is uncalibrated model output, not measured ear-canal pressure.",
    "OHC health is a model parameter, not a fraction of surviving hair cells.",
  ]
  drive = waveform
  if request.middle_ear != "none":
    drive = apply_middle_ear(waveform, FILTERS[request.middle_ear], SAMPLE_RATE)
    warnings.append("The forward human middle-ear filter changes the cochlear input level.")
  if isinstance(request.stimulus, TwoTone):
    ramp_s = request.stimulus.ramp_ms / 1000
    if (
      request.analysis.start_s < ramp_s
      or request.analysis.end_s > request.stimulus.duration_s - ramp_s
    ):
      warnings.append(
        "Analysis includes a stimulus ramp; the spectrum contains onset/offset energy."
      )
  with jax.enable_x64():
    model, simulate, _ = _engine()
    cfs = np.asarray(model.cfs)
    normal = np.ones(model.n_channels)
    impaired = health_profile(request.health, cfs)
    conditions = []
    for index, (identifier, health) in enumerate((("normal", normal), ("impaired", impaired))):
      if progress:
        progress(0.1 + index * 0.4, f"Running {identifier}; first run compiles the model.")
      parameters = model.parameters(jnp.float64)._replace(ohc_health=jnp.asarray(health))
      output = simulate(jnp.asarray(drive, dtype=jnp.float64), parameters)
      output.ear_canal.block_until_ready()
      conditions.append(_condition_result(identifier, output, health, request))
      del output
  result: dict[str, Any] = {
    "kind": "explore",
    "sample_rate": SAMPLE_RATE,
    "duration_s": len(waveform) / SAMPLE_RATE,
    "cfs_hz": cfs.tolist(),
    "analysis_window_s": [request.analysis.start_s, request.analysis.end_s],
    "conditions": conditions,
    "input": _input_preview(waveform, request),
    "warnings": warnings,
    "compute_seconds": time.monotonic() - started,
    "settings": request.model_dump(),
    "model": {
      **_model_metadata(),
      "heatmap": "5 ms RMS; dB re 1 model unit",
      "spectrum": "Hann window; one-sided RMS amplitude; not PSD",
      "waveform_window": "20 ms from analysis start, at original 48 kHz sampling",
    },
  }
  if isinstance(request.stimulus, TwoTone):
    result["markers"] = {
      "f1_hz": request.stimulus.f1_hz,
      "f2_hz": request.stimulus.f2_hz,
      "dp_hz": 2 * request.stimulus.f1_hz - request.stimulus.f2_hz,
    }
  return result


class BenchmarkRequest(Settings):
  """Exploratory observed/model comparison with one shared healthy model anchor."""

  health: Health = Field(default_factory=Health)
  middle_ear: Literal["none", "Verhulst_2018"] = "none"
  point_count: int = Field(default=24, ge=2, le=24)


def run_benchmark(
  request: BenchmarkRequest,
  progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
  """Approximate the observed swept-tone protocol with settled two-tone trials.

  Measured amplitudes remain dB SPL. Model values remain dB re 1 model unit.
  FPL stimulus numbers are used as nominal model SPL, explicitly an
  approximation, with no fitted pressure conversion or pass/fail verdict.
  """
  started = time.monotonic()
  reference = load_reference()
  selected = np.rint(np.linspace(0, len(reference["points"]) - 1, request.point_count)).astype(int)
  points = [reference["points"][int(index)] for index in selected]
  reference["points"] = points
  levels: dict[str, list[float]] = {"normal": [], "impaired": []}
  phases: dict[str, list[float | None]] = {"normal": [], "impaired": []}
  with jax.enable_x64():
    model, _, emit = _engine()
    profiles = [np.ones(model.n_channels), health_profile(request.health, np.asarray(model.cfs))]
    params = [model.parameters(jnp.float64)._replace(ohc_health=jnp.asarray(h)) for h in profiles]
    for index, point in enumerate(points):
      tones = TwoTone(
        f1_hz=point["f1_hz"],
        f2_hz=point["f2_hz"],
        l1_db_spl=reference["stimulus"]["l1_db_fpl"],
        l2_db_spl=reference["stimulus"]["l2_db_fpl"],
        duration_s=1.6,
      )
      run = RunRequest(stimulus=tones, analysis=Analysis(start_s=1.2, end_s=1.5))
      drive = prepare_stimulus(run)
      if request.middle_ear != "none":
        drive = apply_middle_ear(drive, FILTERS[request.middle_ear], SAMPLE_RATE)
      for condition, (name, parameters) in enumerate(
        zip(("normal", "impaired"), params, strict=True)
      ):
        if progress:
          progress(
            (2 * index + condition) / (2 * len(points)),
            f"Frequency {index + 1}/{len(points)}, {name}; DP {point['dp_hz']:.0f} Hz.",
          )
        output = np.asarray(emit(jnp.asarray(drive, dtype=jnp.float64), parameters))
        if not np.isfinite(output).all():
          raise ValueError("Benchmark produced non-finite output.")
        dp = measure_dp(output[57600:72000], 57600, tones)
        levels[name].append(float(20 * np.log10(max(abs(dp), 1e-15))))
        phases[name].append(float(np.angle(dp) / (2 * np.pi)) if abs(dp) > 1e-15 else None)
  observed = np.asarray([point["level_db_spl"] for point in points])
  anchor = levels["normal"][0]
  warnings = list(reference["limitations"]) + [
    "Model stimulus uses 51/36 nominal dB SPL as an approximation to measured 51/36 dB FPL; no FPL-to-SPL conversion is available.",
    "Model phase uses a cosine reference; it is not aligned to the measured phase reference. Phase values below the numerical amplitude floor are omitted.",
    "Relative amplitude uses the lowest frequency: measured curve subtracts its own value; BOTH model conditions subtract the normal model value.",
    "No calibration, parameter fit, or physiological pass/fail is performed.",
  ]
  return {
    "kind": "benchmark",
    "reference": reference,
    "dp_hz": [point["dp_hz"] for point in points],
    "normal_db": levels["normal"],
    "impaired_db": levels["impaired"],
    "normal_phase_cycles": phases["normal"],
    "impaired_phase_cycles": phases["impaired"],
    "reference_relative_db": (observed - observed[0]).tolist(),
    "normal_relative_db": (np.asarray(levels["normal"]) - anchor).tolist(),
    "impaired_relative_db": (np.asarray(levels["impaired"]) - anchor).tolist(),
    "warnings": warnings,
    "compute_seconds": time.monotonic() - started,
    "model": _model_metadata(),
    "settings": {
      **request.model_dump(),
      "sample_rate": SAMPLE_RATE,
      "dtype": "float64",
      "agc_decimation": [1, 1, 1, 1],
      "duration_s": 1.6,
      "analysis_window_s": [1.2, 1.5],
      "ramp_ms": 20,
      "source_coupling": 0.1,
      "reverse_q": 2**-0.5,
      "reverse_middle_ear_gain": 1.0,
      "model_level_units": "dB re 1 model unit (RMS)",
    },
  }
