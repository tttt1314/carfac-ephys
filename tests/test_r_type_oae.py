"""Small analytical checks for the return path and its gain derivative."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from carfac_ephys import r_type_oae as rtype


def test_apical_impulse_crosses_three_lowpasses_with_two_sample_delays():
  samples = 32
  poles = np.array([0.2, 0.5, 0.8], dtype=np.float32)
  coefficients = np.column_stack((1 - poles, np.zeros(3), -poles)).astype(np.float32)
  source = jnp.zeros((samples, 3)).at[0, 2].set(1)

  # Each lowpass has h[n] = (1-p) * p**n. Convolve the three impulse
  # responses, then delay twice to travel from apical channel 2 to basal 0.
  expected = np.array([1.0])
  for pole in poles:
    expected = np.convolve(expected, (1 - pole) * pole ** np.arange(samples))[:samples]
  expected = np.concatenate(([0.0, 0.0], expected[:-2]))
  actual = rtype.backward_path(source, jnp.asarray(coefficients))

  np.testing.assert_array_equal(actual[:2], [0.0, 0.0])
  np.testing.assert_allclose(actual[2:4], [0.08, 0.12], rtol=1e-6)
  np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_beta_scales_rms_pressure_and_has_the_analytical_derivative():
  model = rtype.build_model()
  parameters = rtype.initialize_parameters(model)
  waveform = jnp.zeros(256).at[0].set(0.001)

  @jax.jit
  def pressure(beta_db):
    return rtype.simulate(model, dataclasses.replace(parameters, beta_db=beta_db), waveform)

  def rms_pressure(beta_db):
    return jnp.sqrt(jnp.mean(pressure(beta_db) ** 2))

  # RMS amplitude A(beta) = c * 10**(beta/20), so dA/dbeta = A * ln(10)/20.
  amplitude, derivative = jax.value_and_grad(rms_pressure)(parameters.beta_db)
  assert amplitude > 0
  np.testing.assert_allclose(
    rms_pressure(parameters.beta_db + 20), 10 * amplitude, rtol=1e-5, atol=0
  )
  np.testing.assert_allclose(derivative, amplitude * np.log(10) / 20, rtol=1e-5, atol=0)
