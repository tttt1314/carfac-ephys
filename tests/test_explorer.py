"""Behavior checks for the local CARFAC explorer."""

import numpy as np
import pytest

from carfac_ephys.explorer import RunRequest, prepare_stimulus


def test_two_tone_levels_are_independent_rms_levels():
  request = RunRequest.model_validate(
    {
      "stimulus": {
        "kind": "two_tone",
        "f1_hz": 1000,
        "f2_hz": 1220,
        "l1_db_spl": 65,
        "l2_db_spl": 55,
        "duration_s": 0.4,
      },
      "analysis": {"start_s": 0.1, "end_s": 0.35},
    }
  )
  waveform = prepare_stimulus(request)
  segment = waveform[4800:16800]
  time = np.arange(4800, 16800) / 48000
  for hz, expected in [(1000, 65), (1220, 55)]:
    peak = 2 * abs(np.mean(segment * np.exp(-2j * np.pi * hz * time)))
    level = 104 + 20 * np.log10(peak / np.sqrt(2))
    assert level == pytest.approx(expected, abs=1e-9)
  assert waveform[0] == waveform[-1] == 0


def test_normal_and_health_one_share_same_diagnostics_and_units():
  from carfac_ephys.explorer import run_comparison

  request = RunRequest.model_validate(
    {
      "stimulus": {"kind": "two_tone", "duration_s": 0.06},
      "analysis": {"start_s": 0.02, "end_s": 0.05},
      "health": {"health": 1},
    }
  )
  result = run_comparison(request)
  normal, impaired = result["conditions"]
  assert normal["bm"] == impaired["bm"]
  assert normal["sources"] == impaired["sources"]
  assert normal["spectrum"] == impaired["spectrum"]
  assert np.any(np.asarray(normal["waveforms"]["bm"]) != 0)
  assert np.diff(result["cfs_hz"]).max() < 0
  assert result["markers"]["dp_hz"] == 780
  assert result["sample_rate"] == 48000
  assert result["model"]["output_units"] == "dB re 1 model unit (RMS)"
  assert result["model"]["agc_decimation"] == [1, 1, 1, 1]
  assert len(normal["bm"]["rms_db"]) == len(result["cfs_hz"])


def test_reference_comparison_keeps_impairment_difference_at_shared_anchor():
  from carfac_ephys.explorer import BenchmarkRequest, run_benchmark

  result = run_benchmark(BenchmarkRequest(point_count=2))
  assert len(result["dp_hz"]) == 2
  assert result["normal_relative_db"][0] == pytest.approx(0)
  assert result["reference_relative_db"][0] == pytest.approx(0)
  expected_loss = result["impaired_db"][0] - result["normal_db"][0]
  assert abs(expected_loss) > 0.1
  assert result["impaired_relative_db"][0] == pytest.approx(expected_loss)
  assert result["reference"]["subject"]["health_status"] == "health_not_reported"
  assert result["settings"]["analysis_window_s"] == [1.2, 1.5]
  assert "passed" not in result
