"""Verify source identities, causal propagation, and end-to-end derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from carfac.jax import carfac

from carfac_ephys.oae import DifferentiableOae


@pytest.fixture(scope="module", params=[(8, 2, 2, 2), (1, 1, 1, 1)])
def setup(request):
  model = DifferentiableOae(fs=16000, agc_decimation=request.param)
  parameters = model.parameters()
  t = np.arange(1600) / model.fs
  wave = jnp.asarray(0.02 * np.sin(2 * np.pi * 1000 * t) + 0.01 * np.sin(2 * np.pi * 1230 * t))
  return model, parameters, wave


@pytest.mark.parametrize("decimation", [(8, 2, 2, 2), (1, 1, 1, 1)])
def test_diagnostics_bm_matches_official_carfac_with_impaired_resting_health(decimation):
  with jax.enable_x64():
    model = DifferentiableOae(fs=16000, agc_decimation=decimation)
    parameters = model.parameters(jnp.float64)._replace(
      ohc_health=jnp.linspace(0.25, 1.0, model.n_channels),
      velocity_scale=jnp.asarray(0.13),
      v_offset=jnp.asarray(0.06),
    )
    time = np.arange(1600) / model.fs
    wave = jnp.asarray(
      0.002 + 0.02 * np.sin(2 * np.pi * 1000 * time) + 0.01 * np.sin(2 * np.pi * 1230 * time),
      dtype=jnp.float64,
    )
    diagnostics = jax.jit(model.simulate_diagnostics)(wave, parameters)

    # Independently run the public upstream model from the same silent health.
    design = carfac.CarfacDesignParameters(fs=model.fs)
    design.ears[0].car.use_delay_buffer = True
    design.ears[0].ihc.ihc_style = "two_cap_with_syn"
    design.ears[0].agc.decimation = decimation
    hypers, weights, state = carfac.design_and_init_carfac(design)
    weights.ears[0].car.ohc_health = parameters.ohc_health
    weights.ears[0].car.velocity_scale = float(parameters.velocity_scale)
    weights.ears[0].car.v_offset = float(parameters.v_offset)
    state.ears[0].car.zb_memory = hypers.ears[0].car.zr_coeffs * parameters.ohc_health
    state.ears[0].car.g_memory = carfac.stage_g(weights.ears[0].car, parameters.ohc_health)
    expected = jax.jit(lambda w: carfac.run_segment(w, hypers, weights, state).bm[:, :, 0])(wave)
    assert diagnostics.bm.dtype == jnp.float64
    np.testing.assert_allclose(diagnostics.bm, expected, rtol=1e-10, atol=1e-12)
    # A DC offset makes returning raw_car instead of AC-coupled BM visibly wrong.
    assert np.linalg.norm(diagnostics.bm - diagnostics.sources.raw_car) > 1e-3


@pytest.mark.parametrize("linear_open", [False, True])
def test_diagnostics_preserves_legacy_sources_emission_and_reference_switches(setup, linear_open):
  model, parameters, wave = setup
  matched = model.linearization(parameters)
  reference = matched._replace(radius=matched.radius - 0.02, gain=matched.gain * 0.85)
  switches = {
    "open_loop": linear_open,
    "disable_fast": linear_open,
    "reference": reference,
  }
  diagnostics = jax.jit(lambda w: model.simulate_diagnostics(w, parameters, **switches))(wave)
  legacy = jax.jit(lambda w: model.simulate(w, parameters, **switches))(wave)
  # Existing tuple unpacking remains valid for both legacy public result types.
  sources, output = legacy
  fast, agc, gain, raw_car, mismatch = sources
  for actual, expected in zip(
    diagnostics.sources, (fast, agc, gain, raw_car, mismatch), strict=True
  ):
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-10)
  np.testing.assert_allclose(diagnostics.ear_canal, output, rtol=3e-5, atol=1e-10)
  assert np.linalg.norm(diagnostics.bm) > 0
  assert np.linalg.norm(diagnostics.sources.linear_mismatch) > 0
  if linear_open:
    assert np.max(np.abs(diagnostics.sources.nonlinear)) == 0
    assert np.max(np.abs(diagnostics.ear_canal)) == 0
  else:
    assert np.linalg.norm(diagnostics.sources.nonlinear) > 0
    assert np.linalg.norm(diagnostics.ear_canal) > 0


def test_reference_plus_sources_rebuilds_full_forward_waveform(setup):
  model, parameters, wave = setup
  sources = jax.jit(model.extract_sources)(wave, parameters)
  replay = jax.jit(model.reconstruct_forward)(wave, sources.total, parameters)
  error = np.linalg.norm(np.asarray(replay - sources.raw_car)) / np.linalg.norm(sources.raw_car)
  # Independent full-waveform replay, not a tautological frequency-bin identity.
  assert error < 2e-5
  assert np.linalg.norm(sources.fast) > 0
  assert np.linalg.norm(sources.agc) > 0
  assert np.linalg.norm(sources.gain) > 0


def test_exact_zero_for_linear_open_loop_and_silence(setup):
  model, parameters, wave = setup
  linear = jax.jit(lambda w: model.simulate(w, parameters, open_loop=True, disable_fast=True))(wave)
  quiet = jax.jit(model.simulate)(jnp.zeros_like(wave), parameters)
  assert np.max(np.abs(linear.sources.total)) == 0
  assert np.max(np.abs(linear.ear_canal)) == 0
  assert np.max(np.abs(quiet.ear_canal)) == 0


def test_fast_off_does_not_hide_agc_nonlinearity(setup):
  model, parameters, wave = setup
  sources = jax.jit(lambda w: model.extract_sources(w, parameters, disable_fast=True))(wave)
  assert np.max(np.abs(sources.fast)) == 0
  assert np.linalg.norm(sources.agc) > 0
  assert np.linalg.norm(sources.gain) > 0


def test_no_ohc_health_removes_nonlinear_source_but_keeps_forward_response(setup):
  model, parameters, wave = setup
  parameters = parameters._replace(ohc_health=jnp.zeros(model.n_channels))
  result = jax.jit(model.simulate)(wave, parameters)
  assert np.linalg.norm(result.sources.raw_car) > 0
  assert np.max(np.abs(result.sources.total)) == 0
  assert np.max(np.abs(result.ear_canal)) == 0


@pytest.fixture(scope="module")
def propagation_setup():
  with jax.enable_x64():
    model = DifferentiableOae(fs=16000)
    parameters = model.parameters(jnp.float64)._replace(
      source_coupling=jnp.linspace(0.03, 0.12, model.n_channels),
      middle_ear_gain=jnp.asarray(1.7),
    )
  sources = np.zeros((8192, model.n_channels))
  sources[0, 5] = 1.0
  sources[17:145, [0, 13, 30]] = np.random.default_rng(21).normal(0, 0.02, (128, 3))
  return model, parameters, sources


def _analytic_propagated_spectrum(model, parameters, sources, variant):
  # Evaluate the declared continuous prototypes via s=2fs*(1-z^-1)/(1+z^-1),
  # independently of the implementation's digital coefficients and time scans.
  delay = np.exp(-2j * np.pi * np.arange(len(sources) // 2 + 1) / len(sources))
  minus, plus = 1 - delay, 1 + delay
  rate = 2 * model.fs
  spectrum = np.fft.rfft(sources, axis=0)
  total = np.zeros(len(delay), dtype=complex)
  transfer = np.ones(len(delay), dtype=complex)
  for channel, cf in enumerate(np.asarray(model.cfs)):
    if variant in {"filter_only", "full"}:
      cutoff = rate * np.tan(np.pi * cf / model.fs)
      transfer *= (
        cutoff**2
        * plus**2
        / (
          rate**2 * minus**2
          + cutoff / float(parameters.reverse_q) * rate * minus * plus
          + cutoff**2 * plus**2
        )
      )
    channel_delay = delay**channel if variant in {"delay_only", "full"} else 1
    total += (
      float(parameters.source_coupling[channel]) * transfer * channel_delay * spectrum[:, channel]
    )
  ratio, load, area = 20.0, 1.2 * 343 / 6e-5, 3e-6
  mass, stiffness, damping = 3e-6, 100.0, 0.003
  middle = (
    ratio
    * load
    * area**2
    * rate
    * minus
    * plus
    / (
      mass * rate**2 * minus**2
      + (damping + ratio**2 * load * area**2) * rate * minus * plus
      + stiffness * plus**2
    )
  )
  return float(parameters.middle_ear_gain) * middle * total


@pytest.mark.parametrize("variant", ["direct_sum", "delay_only", "filter_only", "full"])
def test_propagation_variants_match_analytic_multichannel_transfer(propagation_setup, variant):
  model, parameters, sources = propagation_setup
  with jax.enable_x64():
    output = jax.jit(lambda source: model.propagate_sources(source, parameters, variant=variant))(
      jnp.asarray(sources)
    )
  expected = _analytic_propagated_spectrum(model, parameters, sources, variant)
  actual = np.fft.rfft(output)
  assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-9


def test_full_variant_preserves_default_propagation(propagation_setup):
  model, parameters, sources = propagation_setup
  with jax.enable_x64():
    wave = jnp.asarray(sources)
    legacy = jax.jit(model.propagate_sources)(wave, parameters)
    explicit = jax.jit(lambda source: model.propagate_sources(source, parameters, variant="full"))(
      wave
    )
  np.testing.assert_array_equal(explicit, legacy)


@pytest.mark.parametrize("variant", ["direct_sum", "delay_only", "filter_only", "full"])
def test_propagation_variants_are_causal_with_declared_hop_delays(propagation_setup, variant):
  model, parameters, _ = propagation_setup
  onset, channel = 3, 7
  with jax.enable_x64():
    source = jnp.zeros((256, model.n_channels)).at[onset, channel].set(1.0)
    output = jax.jit(lambda source: model.propagate_sources(source, parameters, variant=variant))(
      source
    )
  expected_onset = onset + (channel if variant in {"delay_only", "full"} else 0)
  np.testing.assert_array_equal(output[:expected_onset], 0)
  assert abs(float(output[expected_onset])) > 0


@pytest.mark.parametrize("variant", ["direct_sum", "delay_only", "filter_only", "full"])
def test_propagation_variants_emit_zero_without_sources_or_coupling(propagation_setup, variant):
  model, parameters, sources = propagation_setup
  with jax.enable_x64():
    propagate = jax.jit(lambda source, p: model.propagate_sources(source, p, variant=variant))
    source = jnp.asarray(sources[:256])
    silent = propagate(jnp.zeros_like(source), parameters)
    uncoupled = propagate(source, parameters._replace(source_coupling=jnp.zeros(model.n_channels)))
  np.testing.assert_array_equal(silent, 0)
  np.testing.assert_array_equal(uncoupled, 0)


def test_reverse_impulse_matches_analytic_chain_and_propagation_delay(setup):
  model, parameters, _ = setup
  count, channel = 4096, 12
  impulse = jnp.zeros((count, model.n_channels)).at[0, channel].set(1.0)
  output = np.asarray(jax.jit(model.propagate_sources)(impulse, parameters))
  assert np.max(np.abs(output[:channel])) == 0
  assert np.max(np.abs(output[channel:])) > 0
  delay = np.exp(-2j * np.pi * np.arange(count // 2 + 1) / count)
  b, a = model.reverse_coefficients(parameters)
  expected = np.full(len(delay), float(parameters.source_coupling[channel]), dtype=complex)
  for i in range(channel + 1):
    stage = (b[0][i] + b[1][i] * delay + b[2][i] * delay**2) / (
      1 + a[0][i] * delay + a[1][i] * delay**2
    )
    expected *= np.asarray(stage)
  expected *= delay**channel
  b, a, gain = model.middle_ear_coefficients(parameters)
  expected *= (
    float(gain) * (b[0] + b[1] * delay + b[2] * delay**2) / (1 + a[0] * delay + a[1] * delay**2)
  )
  measured = np.fft.rfft(output)
  assert np.linalg.norm(measured - expected) / np.linalg.norm(expected) < 5e-5


def test_no_source_coupling_means_no_ear_output(setup):
  model, parameters, wave = setup
  parameters = parameters._replace(source_coupling=jnp.zeros(model.n_channels))
  result = jax.jit(model.simulate)(wave, parameters)
  assert np.linalg.norm(result.sources.total) > 0
  assert np.max(np.abs(result.ear_canal)) == 0


def test_gradient_reaches_upstream_ohc_and_reverse_parameters(setup):
  model, parameters, wave = setup
  channel = int(np.argmin(np.abs(np.asarray(model.cfs) - 1200)))
  parameters = parameters._replace(ohc_health=0.9 * parameters.ohc_health)

  def waveform_for(theta):
    p = parameters._replace(
      velocity_scale=parameters.velocity_scale * jnp.exp(theta[0]),
      v_offset=parameters.v_offset + 0.02 * theta[1],
      ohc_health=parameters.ohc_health.at[channel].add(0.02 * theta[2]),
      source_coupling=parameters.source_coupling * jnp.exp(theta[3]),
      reverse_q=parameters.reverse_q * jnp.exp(theta[4]),
      middle_ear_gain=parameters.middle_ear_gain * jnp.exp(theta[5]),
    )
    return model.simulate(wave, p).ear_canal

  nominal = jax.jit(waveform_for)(jnp.zeros(6))
  scale = jnp.mean(nominal**2)

  def loss(theta):
    output = waveform_for(theta)
    # A full-band waveform loss; no selected DP bin appears in this graph.
    return jnp.mean(output**2) / scale

  loss_jit = jax.jit(loss)
  derivative = np.asarray(jax.jit(jax.grad(loss))(jnp.zeros(6)))
  step = 0.003
  finite_difference = []
  for index in range(6):
    perturbation = jnp.zeros(6).at[index].set(step)
    finite_difference.append(float((loss_jit(perturbation) - loss_jit(-perturbation)) / (2 * step)))
  np.testing.assert_allclose(derivative, finite_difference, rtol=0.025, atol=0.002)
  assert np.isfinite(derivative).all()
  assert (np.abs(derivative) > 1e-5).all()


def test_constant_ohc_law_has_no_emission_with_nonzero_forward_response():
  # The upstream linear-CAR law is N(v)=1, different from 1/(1+offset**2).
  model = DifferentiableOae(fs=16000, linear_car=True)
  parameters = model.parameters()._replace(v_offset=jnp.asarray(0.8))
  wave = jnp.asarray(np.random.default_rng(7).normal(0, 0.01, 512))
  result = jax.jit(lambda w: model.simulate(w, parameters, open_loop=True))(wave)
  assert np.linalg.norm(result.sources.raw_car) > 0
  assert np.max(np.abs(result.sources.total)) == 0
  assert np.max(np.abs(result.ear_canal)) == 0
  replay = model.reconstruct_forward(wave, result.sources.total, parameters)
  assert (
    np.linalg.norm(replay - result.sources.raw_car) / np.linalg.norm(result.sources.raw_car) < 5e-5
  )


def test_arbitrary_linear_reference_changes_diagnostic_not_emission():
  model = DifferentiableOae(fs=16000)
  p = model.parameters()._replace(velocity_scale=jnp.asarray(0.0), v_offset=jnp.asarray(0.7))
  wave = jnp.asarray(np.random.default_rng(8).normal(0, 0.01, 800))
  matched = model.linearization(p)
  reference = matched._replace(radius=matched.radius - 0.03, gain=matched.gain * 0.8)
  result = jax.jit(lambda w: model.simulate(w, p, open_loop=True, reference=reference))(wave)
  assert np.max(np.abs(result.sources.nonlinear)) == 0
  assert np.max(np.abs(result.ear_canal)) == 0
  assert np.linalg.norm(result.sources.linear_mismatch) > 0
  replay = model.reconstruct_forward(
    wave, result.sources.reference_residual, p, reference=reference
  )
  assert (
    np.linalg.norm(replay - result.sources.raw_car) / np.linalg.norm(result.sources.raw_car) < 5e-5
  )


def test_closed_loop_source_has_zero_first_order_input_response(setup):
  model, parameters, wave = setup

  def response(amplitude):
    return model.simulate(amplitude * wave, parameters).sources.nonlinear

  value, tangent = jax.jit(lambda: jax.jvp(response, (jnp.asarray(0.0),), (jnp.asarray(1.0),)))()
  assert np.max(np.abs(value)) == 0
  assert np.max(np.abs(tangent)) < 1e-7
  assert np.linalg.norm(response(jnp.asarray(1.0))) > 0


def test_nonlinear_emission_and_gradients_are_independent_of_diagnostic_reference(setup):
  model, parameters, wave = setup
  matched = model.linearization(parameters)

  def simulate(shift):
    reference = matched._replace(radius=matched.radius - shift, gain=matched.gain * 0.85)
    return model.simulate(wave, parameters, reference=reference)

  result = jax.jit(simulate)(jnp.asarray(0.02))
  default = jax.jit(model.simulate)(wave, parameters)
  # Different JIT graphs may fuse float32 arithmetic differently.
  for actual, expected in (
    (result.ear_canal, default.ear_canal),
    (result.sources.nonlinear, default.sources.nonlinear),
  ):
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-5
  assert np.linalg.norm(result.sources.nonlinear) > 0
  assert np.linalg.norm(result.sources.linear_mismatch) > 0
  reference = matched._replace(radius=matched.radius - 0.02, gain=matched.gain * 0.85)
  replay = model.reconstruct_forward(
    wave, result.sources.reference_residual, parameters, reference=reference
  )
  assert (
    np.linalg.norm(replay - result.sources.raw_car) / np.linalg.norm(result.sources.raw_car) < 5e-5
  )
  derivative = jax.jit(jax.grad(lambda shift: jnp.sum(simulate(shift).ear_canal ** 2)))(
    jnp.asarray(0.02)
  )
  assert float(derivative) == 0


def test_full_band_emission_generates_dp_above_single_tone_controls():
  # Frequency labels are used only to construct stimuli and inspect the output.
  with jax.enable_x64():
    model = DifferentiableOae(fs=16000, agc_decimation=(1, 1, 1, 1))
    parameters = model.parameters(jnp.float64)
    time = np.arange(25600) / model.fs
    envelope = np.ones(len(time))
    envelope[:320] = 0.5 - 0.5 * np.cos(np.pi * np.arange(320) / 320)
    tone1 = 0.03 * np.sin(2 * np.pi * 1000 * time) * envelope
    tone2 = 0.015 * np.sin(2 * np.pi * 1230 * time) * envelope
    predict = jax.jit(lambda wave: model.simulate(wave, parameters).ear_canal)

    def dp_amplitude(wave):
      segment = np.asarray(wave)[19200:24000]
      spectrum = 2 * np.fft.rfft(segment) / len(segment)
      return abs(spectrum[231])  # 770 Hz, inspected after full waveform generation.

    dual = dp_amplitude(predict(jnp.asarray(tone1 + tone2)))
    control = max(
      dp_amplitude(predict(jnp.asarray(tone1))), dp_amplitude(predict(jnp.asarray(tone2)))
    )
    assert dp_amplitude(tone1 + tone2) < 1e-12
    assert dual > 1e-7
    assert dual > 10000 * max(control, 1e-15)
