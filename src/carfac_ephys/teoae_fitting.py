"""Fixed-recipe TEOAE fitting and post-fit gain evaluation for thesis §3.4.1."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from carfac.jax import carfac

from carfac_ephys.r_type_oae import (
  PRESSURE_REFERENCE_PA,
  RTypeOAEModel,
  RTypeOAEParameters,
  _simulate_normalized,
  car_weights,
  click_prediction,
  constrain_health,
  initialize_parameters,
)
from carfac_ephys.r_type_oae import (
  THESIS_LOSS_WEIGHTS as _THESIS_LOSS_WEIGHTS,
)

THESIS_LOSS_WEIGHTS = dict(_THESIS_LOSS_WEIGHTS)
STARTS = {"health": 0.99, "slight": 0.75, "mid": 0.5}
PROBE_HZ = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
METRIC_NAMES = ("audiogram_mae_db", "time_rmse_mpa", "spectrum_mae_db")


@dataclasses.dataclass(frozen=True)
class StartResult:
  """One complete initial-state trajectory, including both loss conventions."""

  name: str
  parameters: RTypeOAEParameters
  losses: np.ndarray
  final_recomputed_loss: float

  @property
  def selection_loss(self) -> float:
    """Returns the last pre-update loss used by the original start selection."""
    return float(self.losses[-1])


@dataclasses.dataclass(frozen=True)
class FitResult:
  """All three trajectories and the winner chosen exclusively by TEOAE loss."""

  starts: tuple[StartResult, ...]
  selected: StartResult

  @property
  def parameters(self) -> RTypeOAEParameters:
    """Returns the winning parameters after the final Adam update."""
    return self.selected.parameters


def prepare_target(oae_pa: np.ndarray, model: RTypeOAEModel) -> jax.Array:
  """Normalizes an aligned pressure waveform for the fixed fitting window."""
  target = np.asarray(oae_pa)
  samples = model.config.te_end - model.config.te_start
  if target.shape != (samples,) or not np.isfinite(target).all():
    raise ValueError(f"Expected a finite ({samples},) aligned OAE waveform in Pa.")
  # Normalize before float32 conversion to retain the original arithmetic.
  target = target / PRESSURE_REFERENCE_PA
  if not np.any(target):
    raise ValueError("The original STFT loss requires a nonzero target.")
  return jnp.asarray(target, dtype=jnp.float32)


def multiresolution_stft_loss(prediction: jax.Array, target: jax.Array) -> jax.Array:
  """Computes the author's three-resolution spectral-convergence/log loss."""
  total = 0.0
  for fft_size, hop, window in ((512, 256, 512), (256, 128, 256), (128, 30, 128)):
    pred_mag = jnp.abs(
      jax.scipy.signal.stft(prediction, nperseg=window, noverlap=window - hop, nfft=fft_size)[2]
    )
    target_mag = jnp.abs(
      jax.scipy.signal.stft(target, nperseg=window, noverlap=window - hop, nfft=fft_size)[2]
    )
    convergence = jnp.linalg.norm(target_mag - pred_mag, ord="fro") / jnp.linalg.norm(
      target_mag, ord="fro"
    )
    log_magnitude = jnp.mean(jnp.abs(jnp.log(pred_mag + 1e-7) - jnp.log(target_mag + 1e-7)))
    total = total + (convergence + log_magnitude)
  return jnp.asarray(total / 3)


@jax.jit
def _gaussian_shape_loss(roughness, theoretical):
  return jnp.mean((jnp.sort(roughness) - theoretical) ** 2)


def _loss_terms(parameters, weights, target_data, model):
  cfg = model.config
  target = target_data["target"]
  health = constrain_health(parameters.h_logit)
  updated_car = dataclasses.replace(parameters.car, ohc_health=health)
  updated_ear = dataclasses.replace(weights.ears[0], car=updated_car)
  updated_weights = dataclasses.replace(weights, ears=[updated_ear])
  base_input = model.click_input_pa * 10 ** (parameters.alpha_db[0] / 20) * target_data["amp"]
  response = _simulate_normalized(
    model,
    base_input,
    updated_weights,
    parameters.roughness,
    parameters.beta_db[0],
    cfg.prehistory_samples,
  )
  prediction = response[cfg.te_start : cfg.te_end]
  frequency_step = cfg.fs / (cfg.te_end - cfg.te_start)
  bins = slice(int(600 / frequency_step), int(8000 / frequency_step))
  ft = jnp.fft.rfft(target)[bins]
  fp = jnp.fft.rfft(prediction)[bins]
  frequency = jnp.mean((jnp.log10(jnp.abs(ft) + 1e-6) - jnp.log10(jnp.abs(fp) + 1e-6)) ** 2)
  time = jnp.sqrt(jnp.sum((prediction - target) ** 2) / jnp.prod(jnp.asarray(prediction.shape)))
  smooth = jnp.sum(jnp.square(jnp.diff(health)))
  gaussian = _gaussian_shape_loss(parameters.roughness, model.gaussian_quantiles)
  stft = multiresolution_stft_loss(prediction, target)
  return {
    "stft": stft,
    "time": time,
    "freq": frequency,
    "smooth": smooth,
    "rand": gaussian,
  }


def loss_components(
  model: RTypeOAEModel, parameters: RTypeOAEParameters, target: jax.Array
) -> dict[str, jax.Array]:
  """Returns the five original loss terms; no BM labels enter this function."""
  return _loss_terms(
    parameters, model.baseline_weights, {"target": target, "amp": model.click_amplitude}, model
  )


@jax.jit
def _single_loss(parameters, weights, target_data, model):
  terms = _loss_terms(parameters, weights, target_data, model)
  w = model.loss_weights
  return (
    w["stft"] * terms["stft"]
    + w["smooth"] * terms["smooth"]
    + w["rand"] * terms["rand"]
    + w["freq"] * terms["freq"]
    + w["time"] * terms["time"]
  )


@jax.jit
def loss(model: RTypeOAEModel, parameters: RTypeOAEParameters, target: jax.Array) -> jax.Array:
  """Applies Eq. (3.16) using the original one-example vmap/mean AD structure."""
  targets = {"target": target[None, :], "amp": model.click_amplitude[None]}
  values = jax.vmap(_single_loss, in_axes=(None, None, 0, None))(
    parameters, model.baseline_weights, targets, model
  )
  return jnp.mean(values)


def make_optimizer(parameters: RTypeOAEParameters) -> optax.GradientTransformationExtraArgs:
  """Builds the original partitioned Adam and 1500-step cosine schedule."""
  labels = jax.tree_util.tree_map(lambda _: "auxiliary", parameters)
  labels = labels.replace(
    car=dataclasses.replace(
      labels.car,
      ohc_health="health",
      ga_coeffs="frozen",
      gb_coeffs="frozen",
      gc_coeffs="frozen",
      ac_coeff="frozen",
    )
  )
  schedule = optax.cosine_decay_schedule(init_value=0.002, decay_steps=1500, alpha=0.5)
  return optax.multi_transform(
    {"health": optax.adam(0.05), "auxiliary": optax.adam(schedule), "frozen": optax.set_to_zero()},
    labels,
  )


def fit(
  model: RTypeOAEModel,
  target: jax.Array,
  *,
  max_updates: int = 1501,
  progress: Callable[[str, int, float], None] | None = None,
) -> FitResult:
  """Runs all three starts, selecting by the last finite pre-update loss.

  ``target`` is only the normalized TEOAE window; audiogram labels are absent
  from the interface. Reduced budgets are useful for smoke checks. Nonfinite
  trajectories fail explicitly.
  """
  if not 1 <= max_updates <= 1501:
    raise ValueError("max_updates must be between 1 and 1501.")
  target = jnp.asarray(target, dtype=jnp.float32)
  if target.shape != (model.config.te_end - model.config.te_start,):
    raise ValueError("The normalized target must contain exactly 794 samples.")
  if not np.isfinite(np.asarray(target)).all() or not np.any(np.asarray(target)):
    raise ValueError("Target must be finite and nonzero.")
  optimizer = make_optimizer(initialize_parameters(model))
  value_and_grad = jax.value_and_grad(loss, argnums=1)

  @jax.jit
  def update(model, parameters, optimizer_state, target):
    value, gradients = value_and_grad(model, parameters, target)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state)
    return optax.apply_updates(parameters, updates), optimizer_state, value

  final_loss = jax.jit(loss)
  results = []
  for name, health in STARTS.items():
    parameters = initialize_parameters(model, health)
    initial_car = parameters.car
    optimizer_state = optimizer.init(parameters)  # pyrefly: ignore [bad-argument-type]
    history = []
    for iteration in range(max_updates):
      parameters, optimizer_state, value = update(model, parameters, optimizer_state, target)
      scalar = float(value)
      if not np.isfinite(scalar):
        raise FloatingPointError(f"Nonfinite TEOAE loss at {name}, update {iteration}.")
      history.append(scalar)
      if progress is not None and (
        iteration % 100 == 0 or iteration == max_updates - 1 or scalar < 0.5
      ):
        progress(name, iteration, scalar)
      if scalar < 0.5:
        break
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(parameters)):
      raise FloatingPointError(f"Nonfinite parameters after start {name}.")
    for fixed_name in ("ga_coeffs", "gb_coeffs", "gc_coeffs", "ac_coeff"):
      actual = np.asarray(getattr(parameters.car, fixed_name))
      initial = np.asarray(getattr(initial_car, fixed_name), dtype=actual.dtype)
      if not np.array_equal(actual, initial):
        raise FloatingPointError(f"Frozen coefficient {fixed_name} changed in start {name}.")
    recomputed = float(final_loss(model, parameters, target))
    if not np.isfinite(recomputed):
      raise FloatingPointError(f"Nonfinite final recomputed loss after start {name}.")
    results.append(StartResult(name, parameters, np.asarray(history), recomputed))
  selected = min(results, key=lambda result: result.selection_loss)
  return FitResult(tuple(results), selected)


def evaluate(
  model: RTypeOAEModel,
  parameters: RTypeOAEParameters,
  target: jax.Array,
  tl_profile: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
  """Scores TEOAE fidelity and the original post-fit 10-dB BM gain metric."""
  profile = np.asarray(tl_profile, dtype=np.float32)
  if (
    profile.ndim != 2
    or profile.shape[0] != 2
    or not profile.shape[1]
    or not np.isfinite(profile).all()
  ):
    raise ValueError("TL profile must have two finite rows: frequency and gain reduction.")
  cfg = model.config
  t = jnp.arange(int(cfg.fs * 0.1)) / cfg.fs
  amplitude = 10 ** (10 / 20) * 2e-5 * np.sqrt(2)
  probes = jax.vmap(lambda cf: amplitude * jnp.sin(2 * jnp.pi * cf * t))(model.cfs)

  @jax.jit
  def probe_responses(model, weights):
    bm = jax.vmap(carfac.run_segment, in_axes=(0, None, None, None, None))(
      probes, model.hypers, weights, model.initial_state, False
    ).bm
    steady = bm[:, -int(len(t) / 2) :, :, 0]
    return jnp.diag(jnp.sqrt(jnp.mean(steady**2, axis=1)))

  baseline = np.asarray(probe_responses(model, model.baseline_weights))
  fitted = np.asarray(probe_responses(model, car_weights(model, parameters)))
  raw_gain = 20 * np.log10(baseline / (fitted + 1e-12) + 1e-12)
  gain = np.maximum(raw_gain, 0)
  cfs = np.asarray(model.cfs)
  # Preserve the author's two nearest-neighbour target mappings and JAX interp.
  aligned_target = profile[1, np.argmin(np.abs(profile[0, :, None] - cfs[None, :]), axis=0)]
  target_gain = aligned_target[np.argmin(np.abs(cfs[:, None] - PROBE_HZ[None, :]), axis=0)]
  prediction_gain = np.asarray(
    jnp.interp(jnp.asarray(PROBE_HZ), model.cfs[::-1], jnp.asarray(gain[::-1]))
  )
  prediction = np.asarray(jax.jit(click_prediction)(model, parameters))[cfg.te_start : cfg.te_end]
  target_array = np.asarray(target)
  # Metrics keep the original units and frequency-bin endpoints.
  bins = slice(int(600 / (cfg.fs / len(target_array))), int(8000 / (cfg.fs / len(target_array))))
  target_fft = np.fft.rfft(target_array) / len(target_array)
  prediction_fft = np.fft.rfft(prediction) / len(prediction)
  metrics = {
    "audiogram_mae_db": float(np.mean(np.abs(target_gain - prediction_gain))),
    "time_rmse_mpa": float(
      np.sqrt(
        np.mean((prediction * PRESSURE_REFERENCE_PA - target_array * PRESSURE_REFERENCE_PA) ** 2)
      )
      * 1000
    ),
    "spectrum_mae_db": float(
      np.mean(
        np.abs(
          20 * np.log10(np.abs(target_fft[bins])) - 20 * np.log10(np.abs(prediction_fft[bins]))
        )
      )
    ),
  }
  arrays = {
    "cf_hz": cfs,
    "probe_hz": PROBE_HZ,
    "baseline_rms": baseline,
    "fitted_rms": fitted,
    "raw_gain_reduction_db": raw_gain,
    "gain_reduction_db": gain,
    "target_gain_db": target_gain,
    "prediction_gain_db": prediction_gain,
    "target_pa": target_array * PRESSURE_REFERENCE_PA,
    "prediction_pa": prediction * PRESSURE_REFERENCE_PA,
    "frequency_hz": np.fft.rfftfreq(len(target_array), 1 / cfg.fs),
    "target_fft_pa": target_fft * PRESSURE_REFERENCE_PA,
    "prediction_fft_pa": prediction_fft * PRESSURE_REFERENCE_PA,
    "time_s": np.arange(cfg.te_start, cfg.te_end) / cfg.fs,
  }
  if not all(np.isfinite(x).all() for x in arrays.values()) or not all(
    np.isfinite(x) for x in metrics.values()
  ):
    raise FloatingPointError("Evaluation produced nonfinite values.")
  return metrics, {name: np.asarray(value) for name, value in arrays.items()}
