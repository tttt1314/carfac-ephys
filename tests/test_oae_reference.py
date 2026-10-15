"""Check the published reference independently of the simulation backend."""

import math

import pytest

from carfac_ephys.oae_reference import load_reference


def test_load_reference_preserves_published_component_and_observed_amplitude():
  reference = load_reference()
  assert reference["component"] == "dpoae_D"
  assert reference["license"] == "CC BY 4.0"
  assert reference["stimulus"] == {"ratio": 1.16, "l1_db_fpl": 51, "l2_db_fpl": 36}
  # Independent anchors read from the author's fig8.mat, not generated model values.
  first = reference["points"][0]
  assert first["dp_hz"] == pytest.approx(591.73)
  assert first["level_db_spl"] == pytest.approx(10.047)
  assert first["phase_cycles"] == pytest.approx(1.912551)


def test_primary_frequencies_recover_the_observed_dp_without_interpolation():
  reference = load_reference()
  points = reference["points"]
  assert len(points) == 24
  assert all(a["dp_hz"] < b["dp_hz"] for a, b in zip(points, points[1:]))
  for point in points:
    assert all(math.isfinite(value) for value in point.values())
    assert point["f2_hz"] / point["f1_hz"] == pytest.approx(1.16)
    assert 2 * point["f1_hz"] - point["f2_hz"] == pytest.approx(point["dp_hz"])
    assert 0 < point["f1_hz"] < point["f2_hz"] < 24000
  # Source observation 268, followed by the last eligible observation (419).
  assert points[16]["dp_hz"] == pytest.approx(3791.9)
  assert points[16]["level_db_spl"] == pytest.approx(4.6332)
  assert points[16]["phase_cycles"] == pytest.approx(-0.9901)
  assert points[-1]["dp_hz"] == pytest.approx(10800)
  assert points[-1]["level_db_spl"] == pytest.approx(-28.016)
  assert points[-1]["phase_cycles"] == pytest.approx(-1.4429)


def test_reference_discloses_its_reproducible_origin_and_unverified_hearing_status():
  reference = load_reference()
  assert reference["subject"]["count"] == 1
  assert reference["subject"]["health_status"] == "health_not_reported"
  assert reference["comparison_scope"] == "exploratory_relative_amplitude_shape"
  provenance = reference["provenance"]
  assert provenance["source_variable"] == "fig8.dpoae_D"
  assert provenance["source_point_count"] == 454
  assert provenance["eligible_point_count"] == 386
  assert provenance["source_observation_indices_zero_based"] == [
    0,
    17,
    33,
    50,
    67,
    84,
    100,
    117,
    134,
    151,
    167,
    184,
    201,
    218,
    234,
    251,
    268,
    285,
    301,
    318,
    335,
    354,
    373,
    419,
  ]
  source = provenance["artifacts"][0]
  assert source["filename"] == "fig8.mat"
  assert source["sha256"] == "036605383fa92842b810eb86793a971364544773671400a632790a4c8b5ff198"
  assert reference["limitations"]


def test_callers_can_annotate_their_comparison_without_changing_the_reference():
  reference = load_reference()
  reference["points"][0]["level_db_spl"] = 12345
  assert load_reference()["points"][0]["level_db_spl"] == pytest.approx(10.047)
