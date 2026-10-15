"""Compare declared reverse paths against observed distortion-component data."""

import hashlib
import time
from collections.abc import Callable
from datetime import datetime, timezone
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import Field

from carfac_ephys.explorer import (
  SAMPLE_RATE,
  Analysis,
  RunRequest,
  Settings,
  TwoTone,
  _engine,
  _model_metadata,
  measure_dp,
  prepare_stimulus,
)
from carfac_ephys.middle_ear import FILTERS, apply_middle_ear
from carfac_ephys.oae import PropagationVariant
from carfac_ephys.oae_reference import load_distortion_reference

PATHS = (
  {"id": "direct_sum", "label": "直接加總", "description": "無 CF 濾波、無 channel 間延遲"},
  {"id": "delay_only", "label": "只有延遲", "description": "保留每段一個 sample 延遲，無 CF 濾波"},
  {
    "id": "filter_only",
    "label": "只有 CF 濾波",
    "description": "保留 CF 濾波器，移除額外 channel 間延遲",
  },
  {"id": "full", "label": "完整路徑", "description": "CF 濾波器加上每段一個 sample 延遲"},
)


class PathComparisonRequest(Settings):
  """Keep the first comparison fixed apart from grid density and input filter."""

  point_count: int = Field(default=48, ge=8, le=386)
  middle_ear: Literal["none", "Verhulst_2018"] = "none"


def comparison_grid(frequencies_hz: np.ndarray, point_count: int) -> dict[str, Any]:
  """Choose near-log-spaced observed frequencies without looking at amplitudes.

  Four equal log-frequency bands alternate calibration and held-out tests.
  The same indices and split are used for every propagation hypothesis.
  """
  frequencies = np.asarray(frequencies_hz, dtype=np.float64)
  if (
    frequencies.ndim != 1
    or not np.isfinite(frequencies).all()
    or np.any(frequencies <= 0)
    or np.any(np.diff(frequencies) <= 0)
    or not 8 <= point_count <= len(frequencies)
  ):
    raise ValueError("Require increasing positive frequencies and 8 to N observed points.")
  log_f = np.log(frequencies)
  if point_count == len(frequencies):
    selected = set(range(point_count))
  else:
    targets = np.linspace(log_f[0], log_f[-1], point_count)
    selected = set(np.argmin(np.abs(log_f[:, None] - targets), axis=0).tolist())
    while len(selected) < point_count:
      distance = np.min(np.abs(log_f[:, None] - log_f[sorted(selected)]), axis=1)
      selected.add(int(np.argmax(distance)))
  indices = sorted(selected)
  edges = np.exp(np.linspace(log_f[0], log_f[-1], 5))
  edges[0], edges[-1] = frequencies[0], frequencies[-1]
  roles = ["fit", "test", "fit", "test"]
  bins = np.clip(np.searchsorted(edges, frequencies[indices], side="right") - 1, 0, 3)
  return {
    "indices": indices,
    "split": [roles[int(index)] for index in bins],
    "bands": [
      {"low_hz": float(edges[i]), "high_hz": float(edges[i + 1]), "role": roles[i]}
      for i in range(4)
    ],
  }


def score_path(
  model_db: np.ndarray,
  observed_db: np.ndarray,
  split: list[str],
) -> dict[str, Any]:
  """Fit one dB offset on calibration points, then score untouched predictions.

  The fitted offset is an exploratory signal-unit mapping, not a physical
  ear-canal pressure calibration. Positive residuals mean model above data.
  """
  model = np.asarray(model_db, dtype=np.float64)
  observed = np.asarray(observed_db, dtype=np.float64)
  roles = np.asarray(split)
  if (
    model.ndim != 1
    or model.shape != observed.shape
    or model.shape != roles.shape
    or not np.isfinite(model).all()
    or not np.isfinite(observed).all()
    or not np.isin(roles, ["fit", "test"]).all()
    or not (roles == "fit").any()
    or not (roles == "test").any()
  ):
    raise ValueError("Require finite paired levels and both fit and test observations.")
  offset = float(np.mean((observed - model)[roles == "fit"]))
  aligned = model + offset
  residual = aligned - observed
  metrics = {}
  for role in ("fit", "test", "all"):
    errors = residual if role == "all" else residual[roles == role]
    metrics[role] = {
      "count": len(errors),
      "mae_db": float(np.mean(np.abs(errors))),
      "rmse_db": float(np.sqrt(np.mean(errors**2))),
    }
  return {
    "offset_db": offset,
    "aligned_db": aligned.tolist(),
    "residual_db": residual.tolist(),
    "metrics": metrics,
  }


@lru_cache(maxsize=1)
def _path_engine():
  """Compile source extraction once and four independent full-band propagators."""
  model, _, _ = _engine()
  extract = jax.jit(lambda wave, parameters: model.extract_sources(wave, parameters).nonlinear)
  paths = {
    path["id"]: jax.jit(
      partial(model.propagate_sources, variant=cast(PropagationVariant, path["id"]))
    )
    for path in PATHS
  }
  return model, extract, paths


def run_path_comparison(
  request: PathComparisonRequest,
  *,
  progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
  """Run one healthy forward trajectory per frequency and compare four paths.

  Complete time-domain sources are propagated before measuring the output DP.
  Neither extraction nor any reverse path receives primary-tone frequencies.
  Source arrays are reused unchanged and fingerprinted for reproducibility.
  """
  started = time.monotonic()
  reference = load_distortion_reference()
  frequencies = np.array([point["dp_hz"] for point in reference["points"]])
  grid = comparison_grid(frequencies, request.point_count)
  points = [reference["points"][index] for index in grid["indices"]]
  levels: dict[str, list[float]] = {path["id"]: [] for path in PATHS}
  phases: dict[str, list[float | None]] = {path["id"]: [] for path in PATHS}
  floor_counts = dict.fromkeys(levels, 0)
  fingerprints = []
  duration_s, window = 1.6, [1.2, 1.5]
  start, end = [round(value * SAMPLE_RATE) for value in window]
  with jax.enable_x64():
    model, extract, propagators = _path_engine()
    parameters = model.parameters(jnp.float64)
    for index, point in enumerate(points):
      tones = TwoTone(
        f1_hz=point["f1_hz"],
        f2_hz=point["f2_hz"],
        l1_db_spl=reference["stimulus"]["l1_db_fpl"],
        l2_db_spl=reference["stimulus"]["l2_db_fpl"],
        duration_s=duration_s,
      )
      settings = RunRequest(stimulus=tones, analysis=Analysis(start_s=window[0], end_s=window[1]))
      drive = prepare_stimulus(settings)
      if request.middle_ear != "none":
        drive = apply_middle_ear(drive, FILTERS[request.middle_ear], SAMPLE_RATE)
      if progress:
        progress(index / len(points), f"頻點 {index + 1}/{len(points)}：計算共用 source")
      source = extract(jnp.asarray(drive, dtype=jnp.float64), parameters)
      host_source = np.asarray(source)
      if not np.isfinite(host_source).all():
        raise ValueError("CARFAC produced non-finite sources.")
      fingerprints.append(hashlib.sha256(host_source.tobytes(order="C")).hexdigest())
      del host_source
      for number, path in enumerate(PATHS):
        name = path["id"]
        if progress:
          progress(
            (index + (number + 1) / 5) / len(points),
            f"頻點 {index + 1}/{len(points)} · {path['label']}",
          )
        output = np.asarray(propagators[name](source, parameters))
        if not np.isfinite(output).all():
          raise ValueError(f"The {name} path produced non-finite output.")
        dp = measure_dp(output[start:end], start, tones)
        amplitude = abs(dp)
        levels[name].append(float(20 * np.log10(max(amplitude, 1e-15))))
        phases[name].append(float(np.angle(dp) / (2 * np.pi)) if amplitude > 1e-15 else None)
        floor_counts[name] += int(amplitude <= 1e-15)
      del source
  observed = np.array([point["level_db_spl"] for point in points])
  results: list[dict[str, Any]] = [
    {
      **path,
      "raw_db": levels[path["id"]],
      "phase_cycles": phases[path["id"]],
      "numerical_floor_count": floor_counts[path["id"]],
      **score_path(np.array(levels[path["id"]]), observed, grid["split"]),
    }
    for path in PATHS
  ]
  best = min(results, key=lambda path: path["metrics"]["test"]["rmse_db"])
  direct, full = results[0], results[-1]
  source_files = [Path(__file__), Path(__file__).with_name("oae.py")]
  code_hash = hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest()
  return {
    "kind": "path_comparison",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reference": reference,
    "dp_hz": [point["dp_hz"] for point in points],
    "observed_db_spl": observed.tolist(),
    "observed_phase_cycles": [point["phase_cycles"] for point in points],
    "observed_source_indices": [point["source_index_zero_based"] for point in points],
    "source_sha256": fingerprints,
    "split": grid["split"],
    "split_bands": grid["bands"],
    "paths": results,
    "settings": {
      **request.model_dump(),
      "sample_rate": SAMPLE_RATE,
      "dtype": "float64",
      "agc_decimation": [1, 1, 1, 1],
      "ohc_health": 1.0,
      "duration_s": duration_s,
      "analysis_window_s": window,
      "ramp_ms": 20,
      "source_coupling": 0.1,
      "reverse_q": 2**-0.5,
      "reverse_middle_ear_gain": 1.0,
      "reverse_middle_ear_transfer": "fixed_loaded_stapes",
      "reference_observation_count": len(frequencies),
      "source_extraction_count": len(points),
      "sampling": "Near-log-spaced original observations; no observed interpolation.",
      "split_rule": "Four equal log-frequency bands, ordered fit/test/fit/test.",
      "alignment": "One additive dB offset per path, estimated only from fit bands.",
      "fitted_parameters": ["global_db_offset_per_path"],
      "phase_scored": False,
      "reflection": False,
    },
    "model": {**_model_metadata(), "comparison_code_sha256": code_hash},
    "conclusion": {
      "best_path": best["id"],
      "full_minus_direct_test_rmse_db": (
        full["metrics"]["test"]["rmse_db"] - direct["metrics"]["test"]["rmse_db"]
      ),
    },
    "warnings": [
      *reference["limitations"],
      "The measured swept protocol is approximated by independent settled two-tone trials.",
      "Measured 51/36 dB FPL is used as nominal model SPL, without an FPL-to-SPL calibration.",
      "Global dB offsets align signal units for shape comparison; they are not pressure calibration.",
      "Only calibration bands determine offsets; held-out bands never fit any parameter.",
      "All four paths reuse the identical full-band time-domain source at each frequency.",
      "Model cosine-reference phase is not aligned to measured phase; no phase error is scored.",
      "Fixed healthy model parameters are not a diagnosis of the observed ear's hearing status.",
      "This single-ear exploratory comparison is not a physiological pass/fail test.",
      *(
        [
          "Some model DP values reached the numerical floor; aligned errors at those points are floor-limited."
        ]
        if any(floor_counts.values())
        else []
      ),
    ],
    "compute_seconds": time.monotonic() - started,
  }
