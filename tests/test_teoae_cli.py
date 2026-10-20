import json
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from carfac_ephys import teoae_cli, teoae_fitting


def test_input_preflight_detects_corrupted_archive(tmp_path, monkeypatch):
  folder = tmp_path / "data" / "Flat00"
  folder.mkdir(parents=True)
  oae = folder / "oae.npy"
  profile = folder / "profile.txt"
  np.save(oae, np.ones((6, 2205)))
  profile.write_text("250 500\n0 0\n")
  manifest = tmp_path / "manifest.json"
  manifest.write_text(
    json.dumps(
      {
        "profiles": {
          "Flat00": {
            "oae": {"sha256": teoae_cli.file_sha256(oae), "source": "source/oae.npy"},
            "profile": {"sha256": teoae_cli.file_sha256(profile), "source": "source/profile.txt"},
          }
        }
      }
    )
  )
  monkeypatch.setattr(teoae_cli, "MANIFEST_PATH", manifest)
  assert "Flat00" in teoae_cli.validate_inputs(folder.parent, ("Flat00",))
  profile.write_text("250 500\n0 1\n")
  with pytest.raises(ValueError, match="hash mismatch"):
    teoae_cli.validate_inputs(folder.parent, ("Flat00",))
  result = CliRunner().invoke(
    teoae_cli.main, ["--data-dir", str(folder.parent), "--profiles", "Flat00"]
  )
  assert result.exit_code != 0
  assert "hash mismatch" in result.output


def test_reduced_budget_or_missing_patterns_cannot_pass(tmp_path):
  records: list[dict[str, Any]] = []
  for profile in teoae_fitting.PROFILES:
    folder = tmp_path / profile
    folder.mkdir()
    np.savez(
      folder / "evaluation.npz",
      probe_hz=teoae_fitting.PROBE_HZ,
      target_gain_db=np.zeros(8),
      prediction_gain_db=np.zeros(8),
    )
    metrics = dict.fromkeys(teoae_fitting.METRIC_NAMES, 0.0)
    records.append(
      {
        "profile": profile,
        "metrics": metrics,
        "acceptance": teoae_fitting.acceptance(profile, metrics),
        "selected_start": "health",
        "selection_loss": 1.0,
        "final_recomputed_loss": 1.0,
      }
    )
  assert not teoae_cli.write_report(tmp_path, records, False)["reproduction_passed"]
  assert not teoae_cli.write_report(tmp_path, records[:1], True)["reproduction_passed"]
  assert teoae_cli.write_report(tmp_path, records, True)["reproduction_passed"]
  records[0]["acceptance"]["passed"] = False
  assert not teoae_cli.write_report(tmp_path, records, True)["reproduction_passed"]
