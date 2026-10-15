"""Exercise the local explorer through its public HTTP boundary."""

import hashlib
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from carfac_ephys.explorer_api import create_app


def test_health_and_config_expose_fixed_model_settings():
  with TestClient(create_app()) as client:
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/config")
    assert response.status_code == 200
    settings = response.json()
    assert settings["sample_rate"] == 48000
    assert settings["max_excerpt_s"] == 3
    assert settings["model"]["dtype"] == "float64"
    assert settings["model"]["agc_decimation"] == [1, 1, 1, 1]
    assert settings["defaults"]["stimulus"]["kind"] == "two_tone"


def _wav_bytes(waveform, sample_rate, subtype="FLOAT"):
  buffer = io.BytesIO()
  sf.write(buffer, waveform, sample_rate, format="WAV", subtype=subtype)
  return buffer.getvalue()


def test_audio_upload_and_excerpt_preserve_channel_and_exact_ramped_rms():
  from carfac_ephys.explorer import AudioStimulus
  from carfac_ephys.explorer_api import prepare_audio_excerpt

  sample_rate = 8000
  time = np.arange(round(0.2 * sample_rate)) / sample_rate
  content = _wav_bytes(
    np.column_stack((np.sin(2 * np.pi * 900 * time), np.sin(2 * np.pi * 1300 * time))), sample_rate
  )
  with TestClient(create_app()) as client:
    response = client.post("/api/audio", files={"file": ("stereo.wav", content, "audio/wav")})
    assert response.status_code == 200
    metadata = response.json()
    assert metadata["name"] == "stereo.wav"
    assert metadata["sample_rate"] == sample_rate
    assert metadata["channels"] == 2
    assert metadata["duration_s"] == pytest.approx(0.2)
    settings = AudioStimulus(
      audio_id=metadata["audio_id"], start_s=0.05, duration_s=0.06, level_db_spl=65, channel="right"
    )
    wave = prepare_audio_excerpt(content, settings)
    assert wave.dtype == np.float64
    assert wave.shape == (2880,)
    assert wave[0] == wave[-1] == 0
    assert 104 + 20 * np.log10(np.sqrt(np.mean(wave**2))) == pytest.approx(65, abs=1e-10)
    frequencies = np.fft.rfftfreq(len(wave), 1 / 48000)
    assert frequencies[np.argmax(abs(np.fft.rfft(wave)))] == pytest.approx(1300)


def _wait_for_run(client, identifier, timeout=90):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    response = client.get(f"/api/runs/{identifier}")
    assert response.status_code == 200
    job = response.json()
    assert 0 <= job["progress"] <= 1
    if job["status"] in {"complete", "error"}:
      return job
    time.sleep(0.05)
  raise AssertionError("Simulation did not finish within the test deadline.")


def test_short_real_run_completes_and_exports_matching_health_one():
  request = {
    "stimulus": {"kind": "two_tone", "duration_s": 0.06},
    "analysis": {"start_s": 0.02, "end_s": 0.05},
    "health": {"health": 1},
  }
  with TestClient(create_app()) as client:
    accepted = client.post("/api/runs", json=request)
    assert accepted.status_code == 202
    identifier = accepted.json()["id"]
    assert client.post("/api/runs", json=request).status_code == 409
    job = _wait_for_run(client, identifier)
    assert job["status"] == "complete", job
    assert job["progress"] == 1
    result = job["result"]
    normal, impaired = result["conditions"]
    assert normal["bm"] == impaired["bm"]
    assert normal["spectrum"] == impaired["spectrum"]
    assert result["sample_rate"] == 48000
    assert result["model"]["dtype"] == "float64"
    exported = client.get(f"/api/runs/{identifier}/export")
    assert exported.status_code == 200
    assert exported.json() == result
    assert "attachment" in exported.headers["content-disposition"]
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert (
      client.post(
        "/api/runs", json={**request, "analysis": {"start_s": 0.05, "end_s": 0.04}}
      ).status_code
      == 422
    )


def test_nonfinite_request_is_a_json_validation_error():
  with TestClient(create_app()) as client:
    response = client.post(
      "/api/runs",
      content='{"stimulus":{"kind":"two_tone","f1_hz":NaN}}',
      headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "finite_number"


def test_upload_limits_and_invalid_audio_are_rejected_before_simulation():
  with TestClient(create_app()) as client:
    for content in (
      b"not an audio container",
      _wav_bytes(np.zeros((800, 3)), 8000),
      _wav_bytes(np.zeros(800), 4000),
      _wav_bytes(np.zeros(601 * 8000), 8000, "PCM_16"),
    ):
      response = client.post("/api/audio", files={"file": ("bad.wav", content)})
      assert response.status_code == 422
    response = client.post("/api/audio", files={"file": ("large.wav", b"x" * (16 * 1024**2 + 1))})
    assert response.status_code == 413


@pytest.mark.parametrize("problem", ["silent", "nonfinite", "crop", "right"])
def test_invalid_audio_excerpt_never_starts_a_job(problem):
  samples = np.ones(1600)
  if problem == "silent":
    samples[:] = 0
  elif problem == "nonfinite":
    samples[100] = np.nan
  content = _wav_bytes(samples, 8000)
  with TestClient(create_app()) as client:
    metadata = client.post("/api/audio", files={"file": ("mono.wav", content)}).json()
    stimulus = {"kind": "audio", "audio_id": metadata["audio_id"], "duration_s": 0.06}
    if problem == "crop":
      stimulus["start_s"] = 0.18
    elif problem == "right":
      stimulus["channel"] = "right"
    response = client.post(
      "/api/runs", json={"stimulus": stimulus, "analysis": {"start_s": 0.02, "end_s": 0.05}}
    )
    assert response.status_code == 422
    assert "id" not in response.json()


def test_old_uploads_are_evicted_and_flac_is_accepted():
  content = io.BytesIO()
  sf.write(content, np.ones(1600) * 0.1, 8000, format="FLAC")
  with TestClient(create_app()) as client:
    identifiers = []
    for index in range(5):
      response = client.post(
        "/api/audio", files={"file": (f"clip{index}.flac", content.getvalue())}
      )
      assert response.status_code == 200
      identifiers.append(response.json()["audio_id"])
    response = client.post(
      "/api/runs",
      json={
        "stimulus": {"kind": "audio", "audio_id": identifiers[0], "duration_s": 0.06},
        "analysis": {"start_s": 0.02, "end_s": 0.05},
      },
    )
    assert response.status_code == 404


def test_model_errors_are_retained_without_blocking_later_jobs_and_results_are_bounded():
  request = {
    "stimulus": {"kind": "two_tone", "duration_s": 0.06},
    "analysis": {"start_s": 0.02, "end_s": 0.05},
    "health": {"mode": "band", "low_hz": 23000, "high_hz": 23900},
  }
  with TestClient(create_app()) as client:
    identifiers = []
    for _ in range(5):
      accepted = client.post("/api/runs", json=request)
      assert accepted.status_code == 202
      identifier = accepted.json()["id"]
      identifiers.append(identifier)
      result = _wait_for_run(client, identifier)
      assert result["status"] == "error"
      assert "no CARFAC channels" in result["error"]
      assert client.get(f"/api/runs/{identifier}/export").status_code == 409
    assert client.get(f"/api/runs/{identifiers[0]}").status_code == 404
    assert client.get(f"/api/runs/{identifiers[1]}").status_code == 200


def test_reference_route_and_missing_frontend_instructions(tmp_path):
  with TestClient(create_app(frontend_dir=tmp_path / "dist")) as client:
    reference = client.get("/api/reference")
    assert reference.status_code == 200
    assert len(reference.json()["points"]) >= 2
    assert client.post("/api/benchmarks", json={"point_count": 1}).status_code == 422
    response = client.get("/")
    assert response.status_code == 200
    assert "npm run build" in response.text


def test_uploaded_audio_run_exports_input_identity_without_dp_markers_or_file_bytes():
  time_s = np.arange(3200) / 16000
  content = _wav_bytes(np.sin(2 * np.pi * 900 * time_s), 16000)
  with TestClient(create_app()) as client:
    uploaded = client.post("/api/audio", files={"file": ("voice.wav", content)}).json()
    accepted = client.post(
      "/api/runs",
      json={
        "stimulus": {
          "kind": "audio",
          "audio_id": uploaded["audio_id"],
          "start_s": 0.05,
          "duration_s": 0.06,
          "level_db_spl": 55,
        },
        "analysis": {"start_s": 0.02, "end_s": 0.05},
      },
    )
    assert accepted.status_code == 202
    result = _wait_for_run(client, accepted.json()["id"])
    assert result["status"] == "complete", result
    output = result["result"]
    assert output["input_audio"]["sha256"] == hashlib.sha256(content).hexdigest()
    assert output["input_audio"]["name"] == "voice.wav"
    assert "content" not in output["input_audio"]
    assert "markers" not in output
    assert output["settings"]["stimulus"]["level_db_spl"] == 55
    assert output["duration_s"] == pytest.approx(0.06)
    assert "dp_level_db" not in output["conditions"][0]["spectrum"]
