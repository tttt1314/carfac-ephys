"""Differentiable BM-times-roughness reflection path from Jimmy Xu's thesis.

The forward model is the pinned official CARFAC, configured to match the
author's 69-channel implementation. The return chain and middle-ear filters
follow the archived ``utils_carfac`` code, including its operation order.
"""

import dataclasses
import math
from functools import partial

import jax
import jax.numpy as jnp
from carfac.jax import carfac
from flax import struct
from scipy import signal

PRESSURE_REFERENCE_PA = 2e-5
THESIS_LOSS_WEIGHTS = {
  "stft": 5.9165,
  "time": 0.5683,
  "freq": 454.0997,
  "smooth": 188.4442,
  "rand": 79.6671,
}


@dataclasses.dataclass(frozen=True)
class RTypeOAEConfig:
  """Jimmy's fixed synthetic-TEOAE reproduction configuration."""

  fs: int = 44100
  prehistory_samples: int = 128
  te_start: int = 88
  te_end: int = 882
  roughness_seed: int = 42

  def __post_init__(self) -> None:
    if self.fs != 44100:
      raise ValueError("The thesis reproduction preset requires 44100 Hz.")
    if (self.prehistory_samples, self.te_start, self.te_end) != (
      128,
      88,
      882,
    ):
      raise ValueError("The thesis preset requires its original sample windows.")


@struct.dataclass
class RTypeOAEParameters:
  """Original CAR parameter tree plus input, roughness, and reflection gains.

  Alpha and beta are one-element dB arrays, matching the author's optimizer.
  Velocity scale and offset intentionally retain the author's unconstrained
  parameterization. The original fixed CAR leaves remain in the tree and are
  masked by the optimizer, preserving the author's differentiation structure.
  """

  car: carfac.CarWeights
  alpha_db: jax.Array
  roughness: jax.Array
  beta_db: jax.Array

  @property
  def h_logit(self) -> jax.Array:
    """Returns the unconstrained health coordinates."""
    return self.car.ohc_health

  @property
  def velocity_scale(self) -> float:
    """Returns the scalar nonlinearity velocity scale."""
    return self.car.velocity_scale

  @property
  def v_offset(self) -> float:
    """Returns the scalar nonlinearity velocity offset."""
    return self.car.v_offset


@struct.dataclass
class RTypeOAEModel:
  """Read-only design, baseline weights, and resting state for one ear."""

  config: RTypeOAEConfig = struct.field(pytree_node=False)
  hypers: carfac.CarfacHypers = struct.field(pytree_node=False)
  baseline_weights: carfac.CarfacWeights
  initial_state: carfac.CarfacState
  cfs: jax.Array
  reverse_coefficients: jax.Array
  reverse_state: tuple[jax.Array, jax.Array]
  forward_filter: tuple[jax.Array, jax.Array]
  return_filter: tuple[jax.Array, jax.Array]
  click_input_pa: jax.Array
  gaussian_quantiles: jax.Array
  loss_weights: dict[str, float]
  click_amplitude: jax.Array


def lfilter(b: jax.Array, a: jax.Array, waveform: jax.Array) -> jax.Array:
  """Zero-state DF-II-transposed filtering, differentiable through the signal."""
  length = max(len(b), len(a))
  b = jnp.pad(b, (0, length - len(b)))
  a = jnp.pad(a, (0, length - len(a)))
  b, a = b / a[0], a / a[0]

  def step(state, sample):
    output = b[0] * sample + state[0]
    next_state = b[1:] * sample - a[1:] * output + jnp.pad(state[1:], (0, 1))
    return next_state, output

  return jax.lax.scan(step, jnp.zeros(length - 1, dtype=waveform.dtype), waveform)[1]


def _bandpass(low: float, high: float, order: int, fs: int) -> tuple[jax.Array, jax.Array]:
  coefficients = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
  return jnp.asarray(coefficients[0], dtype=jnp.float32), jnp.asarray(
    coefficients[1], dtype=jnp.float32
  )


def build_model(config: RTypeOAEConfig | None = None) -> RTypeOAEModel:
  """Builds the thesis preset without changing CARFAC's package-wide defaults."""
  config = config or RTypeOAEConfig()
  with jax.enable_x64(False):
    design = carfac.CarfacDesignParameters(fs=config.fs)
    ear = design.ears[0]
    ear.car.first_pole_theta = 12000 / (config.fs / 2) * math.pi
    ear.car.min_pole_hz = 100
    ear.car.linear_car = False
    ear.car.use_delay_buffer = True
    ear.ihc.ihc_style = "two_cap"
    ear.agc.decimation = (8, 2, 2, 2)
    hypers, weights, state = carfac.design_and_init_carfac(design)
    weights.ears[0].car.gc_coeffs = weights.ears[0].car.gc_coeffs * 0.94
    cfs = hypers.ears[0].pole_freqs
    if len(cfs) != 69:
      raise RuntimeError("The pinned CARFAC must produce 69 thesis channels.")

    def coefficients(cf):
      pole = jnp.exp(-2.0 * jnp.pi * cf * 1.1 / config.fs)
      return jnp.array([1.0 - pole, 0.0, -pole])

    forward_filter = _bandpass(400, 5000, 2, config.fs)
    return_filter = _bandpass(800, 4000, 1, config.fs)
    click = jnp.zeros(config.prehistory_samples + config.te_end, dtype=jnp.int32)
    click = click.at[config.prehistory_samples : config.prehistory_samples + 5].set(1)
    click = click * (2 * jnp.sqrt(2))
    click_input = lfilter(*forward_filter, click) * PRESSURE_REFERENCE_PA
    quantiles = jnp.linspace(1.0 / (len(cfs) + 1), len(cfs) / (len(cfs) + 1), len(cfs))
    return RTypeOAEModel(
      config,
      hypers,
      weights,
      state,
      cfs,
      jax.vmap(coefficients)(cfs),
      (jnp.zeros((len(cfs), 1)), jnp.zeros(len(cfs))),
      forward_filter,
      return_filter,
      click_input,
      jax.scipy.stats.norm.ppf(quantiles),
      dict(THESIS_LOSS_WEIGHTS),
      jnp.asarray(1e4, dtype=jnp.float32),
    )


def initialize_parameters(model: RTypeOAEModel, health: float = 0.99) -> RTypeOAEParameters:
  """Initializes the original health, seed-42 roughness, and auxiliary leaves."""
  if not 0 < health < 1:
    raise ValueError("Initial health must be strictly between zero and one.")
  h = jnp.full(model.cfs.shape, health, dtype=jnp.float32)
  return RTypeOAEParameters(
    car=dataclasses.replace(
      model.baseline_weights.ears[0].car, ohc_health=jax.scipy.special.logit(h)
    ),
    alpha_db=jnp.array([0.0], dtype=jnp.float32),
    roughness=jax.random.normal(
      jax.random.PRNGKey(model.config.roughness_seed), model.cfs.shape, dtype=jnp.float32
    ),
    beta_db=jnp.array([-50.0], dtype=jnp.float32),
  )


@jax.jit
def constrain_health(health: jax.Array) -> jax.Array:
  """Matches the author's separately staged sigmoid parameterization."""
  return jax.scipy.special.expit(health)


def car_weights(model: RTypeOAEModel, parameters: RTypeOAEParameters) -> carfac.CarfacWeights:
  """Returns official CARFAC weights while preserving all frozen coefficients."""
  ear = model.baseline_weights.ears[0]
  car = dataclasses.replace(
    parameters.car,
    ohc_health=constrain_health(parameters.h_logit),
  )
  return dataclasses.replace(model.baseline_weights, ears=[dataclasses.replace(ear, car=car)])


@jax.jit
def _lowpass_step(sample, memory, coefficients):
  b0, b1, a1 = coefficients
  output = b0 * sample + memory[0]
  return output, jnp.array([b1 * sample - a1 * output])


@jax.jit
def _return_step(state, local, coefficients):
  memory, previous = state
  arriving = jnp.concatenate([previous[1:], jnp.array([0])]) + local
  output, memory = jax.vmap(_lowpass_step)(arriving, memory, coefficients)
  return (memory, output), output


def backward_path(
  source: jax.Array,
  coefficients: jax.Array,
  initial_state: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Returns the basal signal from the author's apical-to-basal DF-II chain.

  ``source`` is (samples, channels), with basal channel zero. Both the filter
  memory and inter-channel delay start at zero for every call.
  """
  if source.ndim != 2 or source.shape[1] != coefficients.shape[0]:
    raise ValueError("Source must have shape (samples, channels) matching the coefficients.")

  def step(state, local):
    return _return_step(state, local, coefficients)

  if initial_state is None:
    initial_state = (
      jnp.zeros((source.shape[1], 1), dtype=source.dtype),
      jnp.zeros(source.shape[1], dtype=source.dtype),
    )
  # Keep the original full-channel scan output before selecting the basal
  # column: its AD staging matters for strict float32 gradient equivalence.
  return jax.lax.scan(step, initial_state, source)[1][:, 0]


@partial(jax.jit, static_argnames=("prehistory_samples",))
def _simulate_normalized(
  model: RTypeOAEModel,
  cochlear_input_pa: jax.Array,
  weights: carfac.CarfacWeights,
  roughness: jax.Array,
  beta_db: jax.Array,
  prehistory_samples: int,
) -> jax.Array:
  output = carfac.run_segment_jit(
    cochlear_input_pa,
    model.hypers,
    weights,
    model.initial_state,
    open_loop=False,
  )
  bm = jnp.squeeze(output.bm)[prehistory_samples:]
  reflected = backward_path(bm * roughness, model.reverse_coefficients, model.reverse_state)
  # Keep the author's division-before-beta order for float32 fitting parity.
  normalized = reflected / PRESSURE_REFERENCE_PA * 10 ** (beta_db / 20)
  normalized = lfilter(*model.return_filter, normalized)
  return lfilter(*model.forward_filter, normalized)


def simulate(
  model: RTypeOAEModel,
  parameters: RTypeOAEParameters,
  waveform_pa: jax.Array,
  *,
  prehistory_samples: int = 0,
) -> jax.Array:
  """Simulates ear-canal pressure, returning a differentiable pressure waveform.

  Input is a one-dimensional acoustic pressure waveform in Pa. Alpha and the
  forward middle ear are applied once. Explicit prehistory is discarded before
  starting the zero-state return path, as in the author's click experiment.
  Every call starts the cochlear and return paths from their resting states.
  """
  waveform_pa = jnp.asarray(waveform_pa, dtype=jnp.float32)
  if waveform_pa.ndim != 1 or not 0 <= prehistory_samples <= len(waveform_pa):
    raise ValueError("Expected a 1-D waveform and a valid prehistory sample count.")
  cochlear_input = lfilter(*model.forward_filter, waveform_pa)
  cochlear_input = cochlear_input * 10 ** (parameters.alpha_db[0] / 20)
  return (
    _simulate_normalized(
      model,
      cochlear_input,
      car_weights(model, parameters),
      parameters.roughness,
      parameters.beta_db[0],
      prehistory_samples,
    )
    * PRESSURE_REFERENCE_PA
  )


def click_prediction(model: RTypeOAEModel, parameters: RTypeOAEParameters) -> jax.Array:
  """Returns the normalized 80-dB click response, preserving author arithmetic."""
  cochlear_input = (
    model.click_input_pa * 10 ** (parameters.alpha_db[0] / 20) * model.click_amplitude
  )
  return _simulate_normalized(
    model,
    cochlear_input,
    car_weights(model, parameters),
    parameters.roughness,
    parameters.beta_db[0],
    model.config.prehistory_samples,
  )
