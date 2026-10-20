"""Exercise user-supplied inputs and a small complete fitting run."""

import hashlib
import json

import numpy as np
import pytest
from click.testing import CliRunner

from carfac_ephys import teoae_cli


def test_inputs_use_custom_names_and_record_actual_hashes(tmp_path):
  folder = tmp_path / "my-click"
  folder.mkdir()
  oae = folder / "oae.npy"
  np.save(oae, np.ones(794))

  first = teoae_cli.validate_inputs(tmp_path, ())
  assert tuple(first) == ("my-click",)
  assert first["my-click"] == {"oae": {"sha256": hashlib.sha256(oae.read_bytes()).hexdigest()}}

  # A user's different input is accepted and gets its own fingerprint.
  np.save(oae, np.full(794, 2.0))
  second = teoae_cli.validate_inputs(tmp_path, ("my-click",))
  assert second["my-click"]["oae"]["sha256"] != first["my-click"]["oae"]["sha256"]
  (folder / "profile.txt").write_text("250 500\n0 0\n")
  assert "profile" in teoae_cli.validate_inputs(tmp_path, ("my-click",))["my-click"]


def test_cli_fits_custom_input_and_saves_all_starts_without_gain_labels(tmp_path):
  folder = tmp_path / "data" / "my-click"
  folder.mkdir(parents=True)
  t = np.arange(794)
  raw = 2e-5 * (8 * np.exp(-t / 160) * np.sin(t * 0.47) + 0.3 * np.cos(t * 0.13))
  np.save(folder / "oae.npy", raw)
  output = tmp_path / "fit"

  result = CliRunner().invoke(
    teoae_cli.main,
    ["--data-dir", str(folder.parent), "--output-dir", str(output), "--max-updates", "1"],
  )
  assert result.exit_code == 0, result.output
  protocol = json.loads((output / "protocol.json").read_text())
  assert protocol["profiles"] == ["my-click"]
  assert set(protocol["inputs"]["my-click"]) == {"oae"}
  report = json.loads((output / "report.json").read_text())
  record = report["results"][0]
  assert record["profile"] == "my-click"
  assert [start["name"] for start in record["starts"]] == ["health", "slight", "mid"]
  for start in record["starts"]:
    assert start["updates"] == 1
    assert (output / "my-click" / f"start-{start['name']}.npz").is_file()
  assert (output / "my-click/selected-parameters.npz").is_file()
  assert "metrics" not in record
  assert not (output / "my-click/evaluation.npz").exists()


@pytest.mark.parametrize(
  ("profiles", "message"),
  [
    ((), "No profile directories"),
    (("missing",), "Missing input folder"),
    (("same", "same"), "only once"),
  ],
)
def test_invalid_input_selection_fails_before_creating_output(tmp_path, profiles, message):
  output = tmp_path / "fit"
  arguments = ["--data-dir", str(tmp_path), "--output-dir", str(output)]
  for profile in profiles:
    arguments.extend(["--profiles", profile])
  result = CliRunner().invoke(teoae_cli.main, arguments)
  assert result.exit_code != 0
  assert message in result.output
  assert not output.exists()
