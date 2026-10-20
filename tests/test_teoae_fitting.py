"""Check fitting rules with simple losses whose updates can be worked out."""

import jax.numpy as jnp
import numpy as np
import optax
import pytest

from carfac_ephys import r_type_oae as rtype
from carfac_ephys import teoae_fitting as fitting


@pytest.fixture(scope="module")
def model():
  return rtype.build_model()


def test_stopping_uses_pre_update_loss_after_applying_the_update(model, monkeypatch):
  # Start at alpha=0. With loss=0.5+alpha and SGD step 0.25:
  # pre-update losses are [0.5, 0.25], saved alpha=-0.5, final loss=0.
  def simple_loss(model, parameters, target):
    return 0.5 + parameters.alpha_db[0]

  monkeypatch.setattr(fitting, "loss", simple_loss)
  monkeypatch.setattr(fitting, "make_optimizer", lambda parameters: optax.sgd(0.25))
  result = fitting.fit(model, jnp.ones(794), max_updates=3)

  assert [start.name for start in result.starts] == ["health", "slight", "mid"]
  for start in result.starts:
    np.testing.assert_array_equal(start.losses, [0.5, 0.25])
    np.testing.assert_array_equal(start.parameters.alpha_db, [-0.5])
    assert start.selection_loss == 0.25
    assert start.final_recomputed_loss == 0.0


def test_winner_uses_pre_update_loss_even_when_final_loss_ranks_differ(model, monkeypatch):
  # Healthy start: loss=1+alpha; other starts: loss=2+4*alpha.
  # One SGD step of 0.1 gives pre-update [1, 2, 2], post-update [0.9, 0.4, 0.4].
  def simple_loss(model, parameters, target):
    healthy = rtype.constrain_health(parameters.h_logit)[0] > 0.9
    return jnp.where(healthy, 1.0, 2.0) + jnp.where(healthy, 1.0, 4.0) * parameters.alpha_db[0]

  monkeypatch.setattr(fitting, "loss", simple_loss)
  monkeypatch.setattr(fitting, "make_optimizer", lambda parameters: optax.sgd(0.1))
  result = fitting.fit(model, jnp.ones(794), max_updates=1)

  np.testing.assert_array_equal([s.selection_loss for s in result.starts], [1, 2, 2])
  np.testing.assert_allclose(
    [s.final_recomputed_loss for s in result.starts], [0.9, 0.4, 0.4], rtol=1e-6
  )
  assert result.selected.name == "health"
  np.testing.assert_allclose(result.parameters.alpha_db, [-0.1], rtol=1e-6)


def test_target_uses_original_row_window_and_normalization(model):
  raw = np.arange(6 * 2205, dtype=np.float64).reshape(6, 2205) / 1e7
  np.testing.assert_array_equal(
    fitting.prepare_target(raw, model), (raw[2, 132:926] / 2e-5).astype(np.float32)
  )
  with pytest.raises(ValueError, match="nonzero"):
    fitting.prepare_target(np.zeros_like(raw), model)
  with pytest.raises(ValueError, match="finite"):
    fitting.prepare_target(np.full_like(raw, np.nan), model)


def test_acceptance_checks_each_metric_and_accepts_improvement():
  names = fitting.METRIC_NAMES
  for profile, reference in fitting.THESIS_METRICS.items():
    limits = np.asarray(reference) + [1, 0.1, 1]
    metrics = dict(zip(names, limits, strict=True))
    assert fitting.acceptance(profile, metrics)["passed"]
    assert fitting.acceptance(profile, dict.fromkeys(names, 0.0))["passed"]
    for key in names:
      bad = {**metrics, key: metrics[key] + 0.001}
      assert not fitting.acceptance(profile, bad)["passed"]
      assert not fitting.acceptance(profile, {**metrics, key: float("nan")})["passed"]
