"""Frequency-blind CARFAC distortion sources and a differentiable reverse path.

The emitted source uses the actual model's analytic resting linearization.
An optional arbitrary linear reference produces a separate diagnostic mismatch;
that mismatch never enters emission. No Fourier bins are selected. The reverse
chain is a declared phenomenological coupling model, not a calibrated cochlear
force or ear-canal pressure model.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from carfac.jax import carfac


class OaeParameters(NamedTuple):
  """Continuous parameters accepted as JAX leaves, including channel arrays."""

  velocity_scale: jax.Array
  v_offset: jax.Array
  ohc_health: jax.Array
  source_coupling: jax.Array
  reverse_q: jax.Array
  middle_ear_gain: jax.Array


class LinearReference(NamedTuple):
  """Per-channel linear CAR coefficients; diagnostic choices never drive emission."""

  radius: jax.Array
  gain: jax.Array


class SourceSignals(NamedTuple):
  """Model-matched sources, separate reference mismatch, and raw CAR output.

  Nonlinear sources include compression at the fundamentals, not just new
  frequencies. linear_mismatch is a local linear transfer difference driven
  by the actual stage input (which may already contain upstream distortion).
  raw_car is the CAR stage output before AC coupling; it is not the official
  CARFAC `bm` output and carries normalized model units.
  """

  fast: jax.Array
  agc: jax.Array
  gain: jax.Array
  raw_car: jax.Array
  linear_mismatch: jax.Array

  @property
  def total(self) -> jax.Array:
    """Compatibility alias for the model-matched nonlinear source."""
    return self.nonlinear

  @property
  def nonlinear(self) -> jax.Array:
    """Return the source relative to the model's own resting linearization."""
    return self.fast + self.agc + self.gain

  @property
  def reference_residual(self) -> jax.Array:
    """Return the diagnostic residual including a chosen reference mismatch."""
    return self.nonlinear + self.linear_mismatch


class OaeSimulation(NamedTuple):
  """Full-band time-domain output and its source decomposition."""

  sources: SourceSignals
  ear_canal: jax.Array


def _biquad_step(carry, values, b, a):
  """Apply a causal direct-form biquad to scalar or channel-vector samples."""
  x1, x2, y1, y2 = carry
  output = b[0] * values + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2
  return (values, x1, output, y1), output


class DifferentiableOae:
  """Augment the official JAX CARFAC with source extraction and reverse filters.

  Channel zero is basal. Each reverse stage receives its local source plus
  the preceding time sample of the output of the next, more apical stage.
  No reverse signal is fed into the forward CARFAC in this first-order model.
  """

  def __init__(
    self,
    fs: int = 32000,
    agc_decimation: tuple[int, ...] = (8, 2, 2, 2),
    *,
    linear_car: bool = False,
  ):
    if fs <= 0:
      raise ValueError("fs must be positive")
    if len(agc_decimation) != 4 or any(value < 1 for value in agc_decimation):
      raise ValueError("agc_decimation must have four positive factors")
    self.fs = fs
    design = carfac.CarfacDesignParameters(fs=fs)
    design.ears[0].car.use_delay_buffer = True
    design.ears[0].car.linear_car = linear_car
    design.ears[0].ihc.ihc_style = "two_cap_with_syn"
    design.ears[0].agc.decimation = agc_decimation
    self.hypers, self.weights, self.initial_state = carfac.design_and_init_carfac(design)
    self.car = self.hypers.ears[0].car
    self.cfs = self.hypers.ears[0].pole_freqs
    self.n_channels = self.hypers.ears[0].n_ch

  def parameters(self, dtype=jnp.float32) -> OaeParameters:
    """Return nominal parameters; positive parameters can be fitted in log space."""
    return OaeParameters(
      jnp.asarray(0.1, dtype=dtype),
      jnp.asarray(0.04, dtype=dtype),
      jnp.ones(self.n_channels, dtype=dtype),
      jnp.full(self.n_channels, 0.1, dtype=dtype),
      jnp.asarray(2**-0.5, dtype=dtype),
      jnp.asarray(1.0, dtype=dtype),
    )

  def _initialize(self, parameters, dtype, disable_fast):
    """Initialize health at rest, keeping all continuous quantities traceable."""

    def cast(value):
      if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
        return jnp.asarray(value, dtype=dtype)
      return value

    weights = jax.tree.map(cast, self.weights)
    state = jax.tree.map(cast, self.initial_state)
    cw = weights.ears[0].car
    cw.velocity_scale = jnp.asarray(0.0, dtype=dtype) if disable_fast else parameters.velocity_scale
    cw.v_offset = parameters.v_offset
    cw.ohc_health = parameters.ohc_health
    zb0 = self.car.zr_coeffs * parameters.ohc_health
    g0 = carfac.stage_g(cw, parameters.ohc_health)
    state.ears[0].car.zb_memory = zb0
    state.ears[0].car.g_memory = g0
    n0 = (
      jnp.ones_like(parameters.v_offset)
      if self.car.linear_car
      else carfac.ohc_nlf(jnp.zeros(self.n_channels, dtype=dtype), cw)
    )
    radius0 = self.car.r1_coeffs + zb0 * n0
    return weights, state, zb0, g0, n0, radius0

  def linearization(self, parameters: OaeParameters) -> LinearReference:
    """Return the analytic zero-input CAR linearization, retaining gradients.

    At rest, the state multiplying radius/control perturbations is zero.
    Their products are higher order; first-order CAR dynamics use N(0),
    resting undamping, and resting gain. No dense Jacobian is constructed.
    """
    _, _, _, gain, _, radius = self._initialize(parameters, parameters.velocity_scale.dtype, False)
    return LinearReference(radius, gain)

  def extract_sources(
    self,
    waveform: jax.Array,
    parameters: OaeParameters,
    *,
    open_loop: bool = False,
    disable_fast: bool = False,
    reference: LinearReference | None = None,
  ) -> SourceSignals:
    """Extract every channel's time-domain sources without tone metadata.

    For q=R z_previous and r0 fixed at rest, the additive state forcing is
    (r-r0)q. It splits into zb*(N(v)-N(0))*q (fast) and
    (zb-zb0)*N(0)*q (AGC). Both are passed through the reference resonator's
    memory before the additional gain residual is added at its output.
    Here r0 and g0 MUST match the actual model's zero-input linearization.
    An optional reference affects only linear_mismatch/reference_residual.
    Closed-loop AGC and gain deviations remain in the nonlinear source: at
    the initialized silent equilibrium they multiply zero CAR state/input,
    so their contribution has no first-order input response.
    """
    dtype = waveform.dtype
    weights, initial, zb0, g0, n0, radius0 = self._initialize(parameters, dtype, disable_fast)
    a = self.car.a0_coeffs
    c = self.car.c0_coeffs
    h = self.car.h_coeffs
    cw = weights.ears[0].car
    auxiliary = jnp.zeros((2, 2, self.n_channels), dtype=dtype)
    diagnostic_state = jnp.zeros((2, self.n_channels), dtype=dtype)

    def step(carry, sample):
      state, residual_state, ref_state = carry
      previous = state.ears[0].car
      zb = previous.zb_memory + previous.dzb_memory
      gain = previous.g_memory + previous.dg_memory
      velocity = previous.z2_memory - previous.za_memory
      nonlinear = jnp.ones_like(velocity) if self.car.linear_car else carfac.ohc_nlf(velocity, cw)
      q1 = a * previous.z1_memory - c * previous.z2_memory
      q2 = c * previous.z1_memory + a * previous.z2_memory
      delta_radius = jnp.stack((zb * (nonlinear - n0), (zb - zb0) * n0))
      force1 = delta_radius * q1
      force2 = delta_radius * q2
      z1, z2 = residual_state[:, 0], residual_state[:, 1]
      new1 = radius0 * (a * z1 - c * z2) + force1
      new2 = radius0 * (c * z1 + a * z2) + force2
      stage_input = jnp.concatenate((sample[None], previous.zy_memory[:-1]))
      # The official step computes the identical actual radius and state update.
      state, _ = carfac.run_sample(open_loop, self.hypers, weights, state, sample[None])
      actual = state.ears[0].car
      output_residual = g0 * h * new2
      gain_residual = (gain - g0) * (stage_input + h * actual.z2_memory)
      mismatch = jnp.zeros(self.n_channels, dtype=dtype)
      if reference is not None:
        ref1, ref2 = ref_state
        ref_state = jnp.stack(
          (
            reference.radius * (a * ref1 - c * ref2) + stage_input,
            reference.radius * (c * ref1 + a * ref2),
          )
        )
        # Both local linear responses receive the SAME actual stage input.
        # actual z2 minus accumulated nonlinear state gives the matched response.
        matched_output = g0 * (stage_input + h * (actual.z2_memory - new2.sum(axis=0)))
        reference_output = reference.gain * (stage_input + h * ref_state[1])
        mismatch = matched_output - reference_output
      signals = (output_residual[0], output_residual[1], gain_residual, actual.zy_memory, mismatch)
      return (state, jnp.stack((new1, new2), axis=1), ref_state), signals

    _, signals = jax.lax.scan(
      jax.checkpoint(step), (initial, auxiliary, diagnostic_state), waveform
    )
    return SourceSignals(*signals)

  def reconstruct_forward(
    self,
    waveform: jax.Array,
    source: jax.Array,
    parameters: OaeParameters,
    *,
    reference: LinearReference | None = None,
  ) -> jax.Array:
    """Replay a linear cascade plus local output residuals.

    Use sources.nonlinear with the default model-matched linearization.
    With an explicit reference, use sources.reference_residual instead.
    """
    _, _, _, gain0, _, radius0 = self._initialize(parameters, waveform.dtype, False)
    if reference is not None:
      radius0, gain0 = reference.radius, reference.gain
    zeros = jnp.zeros(self.n_channels, dtype=waveform.dtype)
    a, c, h = self.car.a0_coeffs, self.car.c0_coeffs, self.car.h_coeffs

    def step(carry, values):
      z1, z2, previous_output = carry
      sample, innovation = values
      u = jnp.concatenate((sample[None], previous_output[:-1]))
      new1 = radius0 * (a * z1 - c * z2) + u
      new2 = radius0 * (c * z1 + a * z2)
      output = gain0 * (u + h * new2) + innovation
      return (new1, new2, output), output

    _, output = jax.lax.scan(step, (zeros, zeros, zeros), (waveform, source))
    return output

  def reverse_coefficients(self, parameters: OaeParameters):
    """Design fixed-CF, unit-DC-gain passive low-pass biquads by bilinear mapping.

    The analog prototype is wc^2/(s^2+wc*s/Q+wc^2). It suppresses frequencies
    above the local CF. This is an explicit propagation hypothesis, not a
    claim that the CARFAC forward transfer can simply be inverted.
    """
    k = jnp.tan(jnp.pi * self.cfs / self.fs)
    normalizer = 1 / (1 + k / parameters.reverse_q + k**2)
    b0 = k**2 * normalizer
    b = (b0, 2 * b0, b0)
    a = (2 * (k**2 - 1) * normalizer, (1 - k / parameters.reverse_q + k**2) * normalizer)
    return b, a

  def middle_ear_coefficients(self, parameters: OaeParameters):
    """Bilinear-transform a reciprocal loaded-stapes reverse transfer.

    H(s)=n*Zec*Ast^2*s/(m*s^2+(d+n^2*Zec*Ast^2)*s+K).
    Generic parameters are declared here; outputs remain normalized, not Pa.
    """
    ratio, z_ec, area = 20.0, 1.2 * 343 / 6e-5, 3e-6
    mass, stiffness, damping = 3e-6, 100.0, 0.003
    effective_damping = damping + ratio**2 * z_ec * area**2
    rate = 2 * self.fs
    denominator = mass * rate**2 + effective_damping * rate + stiffness
    b0 = ratio * z_ec * area**2 * rate / denominator
    b = (b0, 0.0, -b0)
    a = (
      (-2 * mass * rate**2 + 2 * stiffness) / denominator,
      (mass * rate**2 - effective_damping * rate + stiffness) / denominator,
    )
    return b, a, parameters.middle_ear_gain

  def propagate_sources(self, source: jax.Array, parameters: OaeParameters) -> jax.Array:
    """Send complete causal sources toward the base, then through the middle ear.

    B_i=P_i(z)[coupling_i*S_i+z^-1*B_(i+1)], with B_C=0 (no incoming
    apical wave). Every channel is injected once, and channels retain phase.
    The coupling is a free signal-domain parameter, not a physical force unit.
    """
    b, a = self.reverse_coefficients(parameters)
    zeros = jnp.zeros(self.n_channels, dtype=source.dtype)

    def step(carry, sample):
      previous_output = carry[2]
      arriving = jnp.concatenate((previous_output[1:], jnp.zeros(1, dtype=source.dtype)))
      u = parameters.source_coupling * sample + arriving
      state, output = _biquad_step(carry, u, b, a)
      return state, output[0]

    _, basal = jax.lax.scan(step, (zeros, zeros, zeros, zeros), source)
    me_b, me_a, me_gain = self.middle_ear_coefficients(parameters)
    zero = jnp.asarray(0.0, dtype=source.dtype)
    _, ear_canal = jax.lax.scan(
      lambda carry, sample: _biquad_step(carry, sample, me_b, me_a),
      (zero, zero, zero, zero),
      basal,
    )
    return me_gain * ear_canal

  def simulate(
    self,
    waveform: jax.Array,
    parameters: OaeParameters,
    *,
    open_loop: bool = False,
    disable_fast: bool = False,
    reference: LinearReference | None = None,
  ) -> OaeSimulation:
    """Emit only model-matched nonlinear sources; reference is diagnostic only."""
    sources = self.extract_sources(
      waveform, parameters, open_loop=open_loop, disable_fast=disable_fast, reference=reference
    )
    return OaeSimulation(sources, self.propagate_sources(sources.nonlinear, parameters))
