"""Independent signal-analysis checks using analytically known waveforms."""

import numpy as np
import pytest
from pydantic import ValidationError

from carfac_ephys.explorer import (
  SAMPLE_RATE,
  Health,
  RunRequest,
  TwoTone,
  health_profile,
  measure_dp,
  spectrum,
)
from carfac_ephys.oae import DifferentiableOae


@pytest.mark.parametrize("frequency", [0, 1000, SAMPLE_RATE / 2])
def test_spectrum_recovers_known_rms_including_dc_and_nyquist(frequency):
  count = 4800
  time = np.arange(count) / SAMPLE_RATE
  expected_rms = 0.025
  peak = expected_rms if frequency in (0, SAMPLE_RATE / 2) else np.sqrt(2) * expected_rms
  waveform = peak * np.cos(2 * np.pi * frequency * time)
  frequencies, levels = spectrum(waveform)
  index = int(np.argmin(abs(frequencies - frequency)))
  assert frequencies[index] == frequency
  assert 10 ** (levels[index] / 20) == pytest.approx(expected_rms, rel=1e-7)


def test_off_bin_dp_phasor_ignores_strong_primaries_and_keeps_absolute_sample_phase():
  tones = TwoTone(f1_hz=1007.2, f2_hz=1231.4)
  start_sample = 5432
  time = (start_sample + np.arange(1379)) / SAMPLE_RATE
  dp_frequency = 2 * tones.f1_hz - tones.f2_hz
  dp_rms, dp_phase = 2e-5, 0.9
  primaries = np.sqrt(2) * (
    0.4 * np.cos(2 * np.pi * tones.f1_hz * time + 0.3)
    + 0.3 * np.cos(2 * np.pi * tones.f2_hz * time - 0.7)
  )
  waveform = (
    0.08 + primaries + np.sqrt(2) * dp_rms * np.cos(2 * np.pi * dp_frequency * time + dp_phase)
  )
  expected = dp_rms * np.exp(1j * dp_phase)
  assert measure_dp(waveform, start_sample, tones) == pytest.approx(expected, abs=1e-12)
  assert abs(measure_dp(0.08 + primaries, start_sample, tones)) < 1e-12


def test_silence_has_finite_spectral_floor_and_zero_dp():
  waveform = np.zeros(1379)
  frequencies, levels = spectrum(waveform)
  assert len(frequencies) == len(levels)
  assert np.isfinite(levels).all()
  assert np.max(levels) < -250
  assert measure_dp(waveform, 5432, TwoTone()) == 0j


def test_band_health_uses_real_descending_cfs_and_includes_band_edges():
  cfs = np.asarray(DifferentiableOae(fs=SAMPLE_RATE).cfs)
  assert np.all(np.diff(cfs) < 0)
  settings = Health(mode="band", health=0.2, low_hz=float(cfs[20]), high_hz=float(cfs[10]))
  result = health_profile(settings, cfs)
  np.testing.assert_array_equal(result[:10], np.ones(10))
  np.testing.assert_array_equal(result[10:21], np.full(11, 0.2))
  np.testing.assert_array_equal(result[21:], np.ones(len(cfs) - 21))
  np.testing.assert_array_equal(health_profile(Health(health=0), cfs), np.zeros_like(cfs))
  with pytest.raises(ValueError, match="no CARFAC channels"):
    health_profile(Health(mode="band", low_hz=0, high_hz=float(cfs[-1]) / 2), cfs)


@pytest.mark.parametrize(
  "settings",
  [
    {"stimulus": {"kind": "two_tone", "f1_hz": float("nan")}},
    {"stimulus": {"kind": "two_tone", "l1_db_spl": float("inf")}},
    {"stimulus": {"kind": "two_tone", "f1_hz": 1000, "f2_hz": 2000}},
    {"stimulus": {"kind": "two_tone", "ramp_ms": 100, "duration_s": 0.1}},
    {"analysis": {"start_s": 0.1, "end_s": 0.5}},
    {"health": {"mode": "band", "low_hz": 4000, "high_hz": 1000}},
    {"unrecognized_setting": True},
  ],
)
def test_invalid_scientific_settings_are_rejected_before_model_execution(settings):
  with pytest.raises(ValidationError):
    RunRequest.model_validate(settings)


def test_analysis_window_rejects_primaries_that_cannot_be_resolved():
  with pytest.raises(ValueError, match="[Ww]indow|separat"):
    RunRequest.model_validate(
      {
        "stimulus": {"kind": "two_tone", "f1_hz": 1000, "f2_hz": 1000.00001},
        "analysis": {"start_s": 0.1, "end_s": 0.12},
      }
    )


def test_standalone_dp_fit_rejects_a_degenerate_primary_design():
  tones = TwoTone(f1_hz=1000, f2_hz=1000.00001)
  time = np.arange(960) / SAMPLE_RATE
  waveform = np.sqrt(2) * (
    0.1 * np.cos(2 * np.pi * tones.f1_hz * time + 0.3)
    + 0.2 * np.cos(2 * np.pi * tones.f2_hz * time - 0.2)
  )
  with pytest.raises(ValueError, match="identif|condition|separat|rank|resol"):
    measure_dp(waveform, 0, tones)
