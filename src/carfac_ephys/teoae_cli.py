"""Fit R-type TEOAEs supplied by the user and save parameters and metrics."""

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import click
import jax
import numpy as np

from carfac_ephys import r_type_oae, teoae_fitting


def file_sha256(path: Path) -> str:
  """Hashes an input or output file without deserializing its contents."""
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
  """Writes finite, human-readable JSON."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parameter_arrays(parameters: r_type_oae.RTypeOAEParameters) -> dict[str, np.ndarray]:
  """Exports trainable parameters and frozen CAR coefficients."""
  values = {
    name: np.asarray(getattr(parameters, name))
    for name in ("h_logit", "alpha_db", "roughness", "beta_db", "velocity_scale", "v_offset")
  }
  values.update(
    {
      name: np.asarray(getattr(parameters.car, name))
      for name in ("ga_coeffs", "gb_coeffs", "gc_coeffs", "ac_coeff")
    }
  )
  return values


def validate_inputs(data_dir: Path, profiles: tuple[str, ...]) -> dict[str, dict]:
  """Finds input folders and records actual hashes without an archive manifest."""
  if len(set(profiles)) != len(profiles):
    raise ValueError("Each profile may be requested only once.")
  available = {
    folder.name: folder
    for folder in sorted(data_dir.iterdir())
    if folder.is_dir() and (folder / "oae.npy").is_file()
  }
  selected = profiles or tuple(available)
  if not selected:
    raise ValueError("No profile directories containing oae.npy were found.")
  checked = {}
  for profile in selected:
    if profile not in available:
      raise ValueError(f"Missing input folder containing oae.npy: {profile}")
    folder = available[profile]
    checked[profile] = {"oae": {"sha256": file_sha256(folder / "oae.npy")}}
    if (folder / "profile.txt").is_file():
      checked[profile]["profile"] = {"sha256": file_sha256(folder / "profile.txt")}
  return checked


def _runtime() -> dict:
  return {
    "python": platform.python_version(),
    "versions": {
      name: importlib.metadata.version(name)
      for name in ("jax", "jaxlib", "numpy", "scipy", "optax", "flax", "carfac")
    },
    "implementation_sha256": {
      name: file_sha256(Path(__file__).with_name(name))
      for name in ("r_type_oae.py", "teoae_fitting.py", "teoae_cli.py")
    },
  }


def _save_fit(folder: Path, result: teoae_fitting.FitResult) -> list[dict]:
  records = []
  for start in result.starts:
    path = folder / f"start-{start.name}.npz"
    np.savez_compressed(
      path, allow_pickle=False, **parameter_arrays(start.parameters), losses=start.losses
    )
    records.append(
      {
        "name": start.name,
        "updates": len(start.losses),
        "selection_loss": start.selection_loss,
        "final_recomputed_loss": start.final_recomputed_loss,
        "parameters_sha256": file_sha256(path),
      }
    )
  np.savez_compressed(
    folder / "selected-parameters.npz", allow_pickle=False, **parameter_arrays(result.parameters)
  )
  return records


@click.command()
@click.option(
  "--data-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option(
  "--output-dir",
  type=click.Path(path_type=Path, file_okay=False),
  default="output/teoae-fitting",
  show_default=True,
)
@click.option(
  "--profiles",
  multiple=True,
  help="Repeat to select input folders; default: all folders containing oae.npy.",
)
@click.option(
  "--max-updates",
  type=click.IntRange(1, 1501),
  default=1501,
  show_default=True,
  help="Maximum Adam updates per starting point.",
)
def main(data_dir: Path, output_dir: Path, profiles: tuple[str, ...], max_updates: int) -> None:
  """Fits supplied click-response arrays using the fixed R-type training recipe."""
  try:
    checked = validate_inputs(data_dir, profiles)
  except (OSError, ValueError) as error:
    raise click.ClickException(str(error)) from error
  if (output_dir / "protocol.json").exists():
    raise click.ClickException("Output directory already contains a run; choose a new directory.")
  output_dir.mkdir(parents=True, exist_ok=True)
  with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(False):
    model = r_type_oae.build_model()
    write_json(
      output_dir / "protocol.json",
      {
        "config": dataclasses.asdict(model.config),
        "profiles": tuple(checked),
        "starts": teoae_fitting.STARTS,
        "max_updates": max_updates,
        "loss_weights": teoae_fitting.THESIS_LOSS_WEIGHTS,
        "selection": "minimum last finite pre-update TEOAE loss",
        "inputs": checked,
        "runtime": _runtime(),
      },
    )
    records = []
    for profile, inputs in checked.items():
      folder = output_dir / profile
      folder.mkdir(exist_ok=True)
      started = time.perf_counter()

      def progress(start, iteration, value):
        click.echo(f"{profile} {start} update={iteration} loss={value:.6f}")

      try:
        target = teoae_fitting.prepare_target(
          np.load(data_dir / profile / "oae.npy", allow_pickle=False), model
        )
        result = teoae_fitting.fit(model, target, max_updates=max_updates, progress=progress)
        starts = _save_fit(folder, result)
        record: dict[str, Any] = {
          "profile": profile,
          "selected_start": result.selected.name,
          "selection_loss": result.selected.selection_loss,
          "final_recomputed_loss": result.selected.final_recomputed_loss,
          "starts": starts,
          "seconds": time.perf_counter() - started,
        }
        # Gain labels are optional and enter only after all fitting and selection.
        if "profile" in inputs:
          tl_profile = np.loadtxt(data_dir / profile / "profile.txt", dtype=np.float32)
          metrics, arrays = teoae_fitting.evaluate(model, result.parameters, target, tl_profile)
          np.savez_compressed(folder / "evaluation.npz", allow_pickle=False, **arrays)
          record["metrics"] = metrics
        write_json(folder / "result.json", record)
        records.append(record)
        write_json(output_dir / "report.json", {"results": records})
      except (OSError, ValueError, FloatingPointError) as error:
        write_json(folder / "failure.json", {"profile": profile, "error": str(error)})
        raise click.ClickException(str(error)) from error
    click.echo(f"Results: {output_dir / 'report.json'}")
