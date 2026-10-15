"""Exercise real four-path experiments and durable results through HTTP."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from carfac_ephys.explorer_api import create_app


@pytest.mark.parametrize("result", [None, [], "invalid"])
def test_invalid_saved_result_reports_unavailable(tmp_path, result):
  (tmp_path / "path-comparison-latest.json").write_text(
    json.dumps({"id": "broken", "status": "complete", "result": result})
  )
  with TestClient(create_app(results_dir=tmp_path)) as client:
    response = client.get("/api/path-comparisons/latest")
    assert response.status_code == 503
    assert "invalid" in response.json()["detail"]


def test_real_comparison_scores_fixed_sources_and_survives_server_restart(tmp_path):
  with TestClient(create_app(results_dir=tmp_path)) as client:
    assert client.get("/api/path-comparisons/latest").json() is None
    reference = client.get("/api/path-comparisons/reference").json()
    assert len(reference["points"]) == 386
    accepted = client.post("/api/path-comparisons", json={"point_count": 8})
    assert accepted.status_code == 202, accepted.text
    identifier = accepted.json()["id"]
    assert client.post("/api/path-comparisons", json={"point_count": 8}).status_code == 409
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
      job = client.get(f"/api/runs/{identifier}").json()
      if job["status"] in {"complete", "error"}:
        break
      time.sleep(0.05)
    assert job["status"] == "complete", job
    result = job["result"]
    assert result["kind"] == "path_comparison"
    assert len(result["reference"]["points"]) == 386
    assert len(result["dp_hz"]) == len(result["source_sha256"]) == 8
    assert all(len(digest) == 64 for digest in result["source_sha256"])
    assert result["settings"]["ohc_health"] == 1
    assert result["settings"]["source_coupling"] == 0.1
    assert result["settings"]["reverse_middle_ear_gain"] == 1
    assert result["settings"]["source_extraction_count"] == 8
    assert result["settings"]["phase_scored"] is False
    assert {path["id"] for path in result["paths"]} == {
      "direct_sum",
      "delay_only",
      "filter_only",
      "full",
    }
    observed = np.array(result["observed_db_spl"])
    fit = np.array(result["split"]) == "fit"
    for path in result["paths"]:
      raw = np.array(path["raw_db"])
      assert path["offset_db"] == pytest.approx(np.mean((observed - raw)[fit]))
      residual = raw + path["offset_db"] - observed
      np.testing.assert_allclose(path["residual_db"], residual)
      assert path["metrics"]["test"]["rmse_db"] == pytest.approx(
        np.sqrt(np.mean(residual[~fit] ** 2))
      )
      assert len(path["phase_cycles"]) == 8
    best = min(result["paths"], key=lambda path: path["metrics"]["test"]["rmse_db"])
    assert result["conclusion"]["best_path"] == best["id"]
    assert "passed" not in result
    assert client.get("/api/path-comparisons/latest").json()["result"] == result
    assert client.get(f"/api/runs/{identifier}/export").json() == result

  with TestClient(create_app(results_dir=tmp_path)) as restarted:
    saved = restarted.get("/api/path-comparisons/latest").json()
    assert saved["id"] == identifier
    assert saved["result"] == result
    assert restarted.get(f"/api/runs/{identifier}/export").json() == result
