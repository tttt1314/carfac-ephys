"""Validate calibration and prediction through the path comparison boundary."""

import numpy as np
import pytest

from carfac_ephys.oae_reference import load_distortion_reference
from carfac_ephys.path_comparison import comparison_grid, score_path


def test_one_offset_uses_only_fit_points_and_preserves_held_out_error():
  model = np.arange(-100.0, -20.0, 10.0)
  observed = model + 80
  roles = ["fit", "fit", "test", "test", "fit", "fit", "test", "test"]
  observed[[2, 3, 6, 7]] += [3, -4, 0, 5]
  result = score_path(model, observed, roles)
  assert result["offset_db"] == pytest.approx(80)
  np.testing.assert_allclose(result["aligned_db"], model + 80)
  np.testing.assert_allclose(np.array(result["residual_db"])[[2, 3, 6, 7]], [-3, 4, 0, -5])
  assert result["metrics"]["fit"]["rmse_db"] == 0
  assert result["metrics"]["test"]["count"] == 4
  assert result["metrics"]["test"]["mae_db"] == pytest.approx(3)
  assert result["metrics"]["test"]["rmse_db"] == pytest.approx(np.sqrt(12.5))
  observed[[2, 3, 6, 7]] += 100
  changed = score_path(model, observed, roles)
  assert changed["offset_db"] == result["offset_db"]
  assert changed["metrics"]["test"]["rmse_db"] > 90


def test_frequency_grid_uses_observations_and_declares_blocked_split_without_levels():
  reference = load_distortion_reference()
  assert len(reference["points"]) == 386
  frequencies = np.array([point["dp_hz"] for point in reference["points"]])
  grid = comparison_grid(frequencies, 48)
  indices = grid["indices"]
  assert len(indices) == len(set(indices)) == 48
  assert indices[0] == 0 and indices[-1] == len(frequencies) - 1
  assert np.diff(indices).min() > 0
  assert {band["role"] for band in grid["bands"]} == {"fit", "test"}
  assert [band["role"] for band in grid["bands"]] == ["fit", "test", "fit", "test"]
  assert grid["split"].count("fit") >= 20
  assert grid["split"].count("test") >= 20
  for frequency, role in zip(frequencies[indices], grid["split"], strict=True):
    band = (
      next(band for band in grid["bands"] if band["low_hz"] <= frequency < band["high_hz"])
      if frequency != frequencies[-1]
      else grid["bands"][-1]
    )
    assert band["role"] == role
  all_points = comparison_grid(frequencies, len(frequencies))
  assert all_points["indices"] == list(range(len(frequencies)))


def test_path_comparison_request_excludes_health_and_free_path_parameters():
  from pydantic import ValidationError

  from carfac_ephys.path_comparison import PathComparisonRequest

  request = PathComparisonRequest()
  assert request.point_count == 48
  assert request.middle_ear == "none"
  for settings in (
    {"health": {"health": 0.4}},
    {"reverse_q": 5},
    {"point_count": 7},
    {"point_count": 387},
    {"middle_ear": "unrecorded"},
  ):
    with pytest.raises(ValidationError):
      PathComparisonRequest.model_validate(settings)
