"""Reproducible local training, artifacts, and acceptance report for R-type OAE."""

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import click
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import ticker

from carfac_ephys import r_type_oae, teoae_fitting

MANIFEST_PATH = Path(__file__).parent / "data" / "jimmy_teoae_manifest.json"


def file_sha256(path: Path) -> str:
  """Hashes one artifact without deserializing its contents."""
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
  """Writes finite, human-readable JSON."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parameter_arrays(parameters: r_type_oae.RTypeOAEParameters) -> dict[str, np.ndarray]:
  """Exports the six trainable groups and four frozen CAR coefficient arrays."""
  values = {
    name: np.asarray(getattr(parameters, name))
    for name in (
      "h_logit",
      "alpha_db",
      "roughness",
      "beta_db",
      "velocity_scale",
      "v_offset",
    )
  }
  values.update(
    {
      name: np.asarray(getattr(parameters.car, name))
      for name in ("ga_coeffs", "gb_coeffs", "gc_coeffs", "ac_coeff")
    }
  )
  return values


def validate_inputs(data_dir: Path, profiles: tuple[str, ...]) -> dict[str, dict]:
  """Checks every requested raw input against the sealed archival manifest."""
  manifest = json.loads(MANIFEST_PATH.read_text())
  checked = {}
  for profile in profiles:
    checked[profile] = {}
    for kind, filename in (("oae", "oae.npy"), ("profile", "profile.txt")):
      source = data_dir / profile / filename
      expected = manifest["profiles"][profile][kind]
      if not source.is_file():
        raise ValueError(f"Missing {source}. Run scripts/prepare_teoae_data.py first.")
      actual_hash = file_sha256(source)
      if actual_hash != expected["sha256"]:
        raise ValueError(f"Archival input hash mismatch: {source}")
      checked[profile][kind] = {"sha256": actual_hash, "archived_source": expected["source"]}
  return checked


def _runtime() -> dict:
  versions = {
    name: importlib.metadata.version(name)
    for name in ("jax", "jaxlib", "numpy", "scipy", "optax", "flax", "carfac")
  }
  revision = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
  ).stdout.strip()
  return {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "versions": versions,
    "devices": [str(d) for d in jax.devices()],
    "jax_x64_enabled": jax.config.x64_enabled,
    "git_head": revision,
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


def _plot_teoae(folder: Path, profile: str, arrays: dict[str, np.ndarray]) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
  for label, prefix in (("TL target", "target"), ("CARFAC fit", "prediction")):
    axes[0].plot(arrays["time_s"] * 1000, arrays[f"{prefix}_pa"] * 1000, label=label, linewidth=0.9)
    axes[1].plot(
      arrays["frequency_hz"],
      20 * np.log10(np.maximum(np.abs(arrays[f"{prefix}_fft_pa"]), 1e-30)),
      label=label,
    )
  axes[0].set(xlabel="Time after model click (ms)", ylabel="Pressure (mPa)")
  axes[1].set(xlabel="Frequency (Hz)", ylabel="FFT magnitude (dB re 1 Pa)", xlim=(600, 8000))
  for ax in axes:
    ax.legend()
    ax.grid(alpha=0.2)
  fig.suptitle(profile)
  fig.tight_layout()
  fig.savefig(folder / "teoae.png", dpi=160)
  plt.close(fig)


def write_report(output_dir: Path, records: list[dict], full_budget: bool) -> dict:
  """Produces the six-panel gain figure, Table 3.1 comparison, and verdict."""
  complete = tuple(record["profile"] for record in records) == teoae_fitting.PROFILES
  passed = complete and full_budget and all(record["acceptance"]["passed"] for record in records)
  report = {
    "complete_six_patterns": complete,
    "full_recipe_budget": full_budget,
    "reproduction_passed": passed,
    "results": records,
  }
  write_json(output_dir / "report.json", report)
  fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
  by_name = {record["profile"]: record for record in records}
  for ax, profile in zip(axes.flat, teoae_fitting.PROFILES, strict=True):
    ax.set_title(profile)
    if profile in by_name:
      with np.load(output_dir / profile / "evaluation.npz", allow_pickle=False) as arrays:
        ax.semilogx(arrays["probe_hz"], arrays["target_gain_db"], "o-", label="TL target")
        ax.semilogx(arrays["probe_hz"], arrays["prediction_gain_db"], "s-", label="CARFAC fit")
      ax.legend(fontsize=8, loc="lower left")
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain reduction (dB)", ylim=(50, -10))
    ax.set_xticks([250, 1000, 3000, 8000], labels=["250", "1k", "3k", "8k"])
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(alpha=0.2)
  fig.tight_layout()
  fig.savefig(output_dir / "gain-comparison.png", dpi=160)
  plt.close(fig)
  lines = [
    "# Jimmy R-type TEOAE: thesis §3.4.1 reproduction",
    "",
    f"Full six-pattern reproduction: **{'PASS' if passed else 'NOT PASSED'}**.",
    "",
    "Fixed Eq. (3.16) weights; three starts per pattern. Start selection uses only the last pre-update TEOAE loss. "
    "Audiogram labels are used only after selection. Measurements are model gain reductions, not clinical thresholds.",
    "",
    "| Pattern | MAE dB (paper / fit / limit) | Time mPa (paper / fit / limit) | Spectrum dB (paper / fit / limit) | Result |",
    "|---|---|---|---|---|",
  ]
  for record in records:
    a = record["acceptance"]
    cells = [
      f"{a['reference'][key]:.3f} / {record['metrics'][key]:.3f} / {a['limits'][key]:.3f}"
      for key in teoae_fitting.METRIC_NAMES
    ]
    lines.append(
      f"| {record['profile']} | {' | '.join(cells)} | {'PASS' if a['passed'] else 'FAIL'} |"
    )
  lines += [
    "",
    "![Gain comparison](gain-comparison.png)",
    "",
    "## Training and provenance",
    "",
    "The protocol, runtime versions, input SHA-256 hashes and original archive aliases are in [protocol.json](protocol.json). "
    "Each pattern folder contains all three parameter trajectories, loss histories, selected parameters, metrics and evaluation arrays.",
    "",
  ]
  for record in records:
    profile = record["profile"]
    lines += [
      f"### {profile}",
      "",
      f"Selected start: `{record['selected_start']}`. "
      f"Last pre-update loss: {record['selection_loss']:.6f}; recomputed final loss: {record['final_recomputed_loss']:.6f}.",
      "",
      f"![{profile} TEOAE]({profile}/teoae.png)",
      "",
    ]
  lines += [
    "## Interpretation",
    "",
    "Acceptance limits were fixed before fitting: Table 3.1 plus 1 dB audiogram MAE, 0.1 mPa time RMSE, and 1 dB spectrum MAE. "
    "A reduced budget or an incomplete pattern collection cannot pass the reproduction verdict. "
    "The historical thesis fitted checkpoints are unavailable; these are fresh fits to the explicitly identified archived inputs. "
    "No Optuna search or post-hoc loss-weight adjustment was performed.",
    "",
  ]
  (output_dir / "report.md").write_text("\n".join(lines))
  return report


@click.command()
@click.option(
  "--data-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option(
  "--output-dir",
  type=click.Path(path_type=Path, file_okay=False),
  default="output/teoae-thesis",
  show_default=True,
)
@click.option(
  "--profiles",
  multiple=True,
  type=click.Choice(teoae_fitting.PROFILES),
  default=teoae_fitting.PROFILES,
  help="Repeat this option to select patterns; default: all six.",
)
@click.option(
  "--max-updates",
  type=click.IntRange(1, 1501),
  default=1501,
  show_default=True,
  help="Smaller values are smoke runs and cannot pass full reproduction.",
)
def main(data_dir: Path, output_dir: Path, profiles: tuple[str, ...], max_updates: int) -> None:
  """Fits archived synthetic TEOAEs and evaluates the fixed thesis criteria."""
  if len(set(profiles)) != len(profiles):
    raise click.ClickException("Each profile may be requested only once.")
  profiles = tuple(profile for profile in teoae_fitting.PROFILES if profile in profiles)
  try:
    checked = validate_inputs(data_dir, profiles)
  except ValueError as error:
    raise click.ClickException(str(error)) from error
  if (output_dir / "protocol.json").exists():
    raise click.ClickException("Output directory already contains a run; choose a new directory.")
  output_dir.mkdir(parents=True, exist_ok=True)
  cpu = jax.devices("cpu")[0]
  with jax.default_device(cpu), jax.enable_x64(False):
    model = r_type_oae.build_model()
    protocol = {
      "recipe": "Jimmy thesis Eq. (3.16), fresh 3-start TEOAE-only fitting",
      "config": dataclasses.asdict(model.config),
      "profiles": profiles,
      "starts": teoae_fitting.STARTS,
      "max_updates": max_updates,
      "loss_weights": teoae_fitting.THESIS_LOSS_WEIGHTS,
      "selection": "minimum last finite pre-update TEOAE loss",
      "inputs": checked,
      "runtime": _runtime(),
      "source_manifest_sha256": file_sha256(MANIFEST_PATH),
    }
    write_json(output_dir / "protocol.json", protocol)
    snapshot_dir = output_dir / "provenance"
    snapshot_dir.mkdir(exist_ok=True)
    for name, expected in protocol["runtime"]["implementation_sha256"].items():
      shutil.copyfile(Path(__file__).with_name(name), snapshot_dir / name)
      if file_sha256(snapshot_dir / name) != expected:
        raise click.ClickException("Implementation changed while capturing run provenance.")
    records = []
    for profile in profiles:
      folder = output_dir / profile
      folder.mkdir(exist_ok=True)
      target = teoae_fitting.prepare_target(
        np.load(data_dir / profile / "oae.npy", allow_pickle=False), model
      )
      started = time.perf_counter()

      def progress(start, iteration, value):
        click.echo(
          f"{profile} {start} update={iteration} loss={value:.6f} elapsed={time.perf_counter() - started:.1f}s"
        )
        write_json(
          folder / "progress.json", {"start": start, "iteration": iteration, "loss": value}
        )

      try:
        result = teoae_fitting.fit(model, target, max_updates=max_updates, progress=progress)
        starts = _save_fit(folder, result)
        # The gain labels are loaded only once all fitting and start selection end.
        tl_profile = np.loadtxt(data_dir / profile / "profile.txt", dtype=np.float32)
        metrics, arrays = teoae_fitting.evaluate(model, result.parameters, target, tl_profile)
        np.savez_compressed(folder / "evaluation.npz", allow_pickle=False, **arrays)
        record = {
          "profile": profile,
          "selected_start": result.selected.name,
          "selection_loss": result.selected.selection_loss,
          "final_recomputed_loss": result.selected.final_recomputed_loss,
          "starts": starts,
          "seconds": time.perf_counter() - started,
          "metrics": metrics,
          "acceptance": teoae_fitting.acceptance(profile, metrics),
        }
        write_json(folder / "result.json", record)
        _plot_teoae(folder, profile, arrays)
        records.append(record)
        write_report(output_dir, records, max_updates == 1501)
        click.echo(f"{profile}: {metrics}; {'PASS' if record['acceptance']['passed'] else 'FAIL'}")
      except (ValueError, FloatingPointError) as error:
        write_json(folder / "failure.json", {"profile": profile, "error": str(error)})
        raise click.ClickException(str(error)) from error
    report = write_report(output_dir, records, max_updates == 1501)
    click.echo(f"Report: {output_dir / 'report.md'}")
    if max_updates == 1501 and not report["reproduction_passed"]:
      raise click.ClickException("Full reproduction criteria not met; see the saved report.")
