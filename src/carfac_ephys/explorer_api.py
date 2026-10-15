"""Local HTTP boundary and command-line entry point for the CARFAC explorer."""

import hashlib
import html
import io
import json
import math
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import click
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from scipy import signal

from carfac_ephys import constants
from carfac_ephys.explorer import (
  SAMPLE_RATE,
  AudioStimulus,
  BenchmarkRequest,
  RunRequest,
  prepare_stimulus,
  run_benchmark,
  run_comparison,
)
from carfac_ephys.oae_reference import load_distortion_reference, load_reference
from carfac_ephys.path_comparison import PathComparisonRequest, run_path_comparison

MAX_UPLOAD_BYTES = 16 * 1024**2
MAX_UPLOAD_DURATION_S = 600
MAX_UPLOADS = 4
MAX_RESULTS = 4
DEFAULT_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "output" / "explorer"


def _audio_metadata(content: bytes) -> dict[str, Any]:
  """Inspect the container without decoding an unbounded recording."""
  try:
    with sf.SoundFile(io.BytesIO(content)) as recording:
      if recording.format not in {"WAV", "WAVEX", "FLAC"}:
        raise ValueError("Only WAV and FLAC recordings are supported.")
      if recording.channels not in {1, 2}:
        raise ValueError("Recording must have one or two channels.")
      if not 8000 <= recording.samplerate <= 192000:
        raise ValueError("Recording sample rate must be between 8 and 192 kHz.")
      duration = recording.frames / recording.samplerate
      if duration <= 0 or duration > MAX_UPLOAD_DURATION_S:
        raise ValueError("Recording duration must be positive and at most 10 minutes.")
      return {
        "duration_s": duration,
        "sample_rate": recording.samplerate,
        "channels": recording.channels,
      }
  except (RuntimeError, ValueError) as error:
    raise ValueError(f"Cannot use this audio recording: {error}") from error


def prepare_audio_excerpt(content: bytes, stimulus: AudioStimulus) -> np.ndarray:
  """Decode only the requested crop and normalize RMS after cosine ramps.

  A recording retains its DC component. Channel selection precedes anti-aliased
  resampling to 48 kHz; a 20-ms ramp is then applied at either end. Finally the
  entire excerpt RMS is set using the repository's nominal 104-dB reference.
  """
  metadata = _audio_metadata(content)
  if stimulus.start_s + stimulus.duration_s > metadata["duration_s"] + 1e-9:
    raise ValueError("Audio crop extends beyond the recording.")
  if stimulus.channel == "right" and metadata["channels"] != 2:
    raise ValueError("The right channel is unavailable in a mono recording.")
  rate = metadata["sample_rate"]
  count = round(stimulus.duration_s * rate)
  with sf.SoundFile(io.BytesIO(content)) as recording:
    recording.seek(round(stimulus.start_s * rate))
    excerpt = recording.read(count, dtype="float64", always_2d=True)
  if len(excerpt) != count or not np.isfinite(excerpt).all():
    raise ValueError("Audio crop is incomplete or contains non-finite samples.")
  if stimulus.channel == "mix":
    wave = excerpt.mean(axis=1)
  else:
    wave = excerpt[:, int(stimulus.channel == "right")]
  if rate != SAMPLE_RATE:
    divisor = math.gcd(rate, SAMPLE_RATE)
    wave = signal.resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor)
  wanted = round(stimulus.duration_s * SAMPLE_RATE)
  wave = np.asarray(wave[:wanted], dtype=np.float64)
  if len(wave) < wanted:
    wave = np.pad(wave, (0, wanted - len(wave)))
  ramp_count = round(0.020 * SAMPLE_RATE)
  ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp_count))
  wave[:ramp_count] *= ramp
  wave[-ramp_count:] *= ramp[::-1]
  rms = np.sqrt(np.mean(wave**2))
  if not np.isfinite(rms) or rms <= np.finfo(np.float64).tiny:
    raise ValueError("Audio crop is silent or has an invalid RMS level.")
  return wave * (constants.db_spl_to_amplitude(stimulus.level_db_spl) / rms)


def create_app(frontend_dir: Path | None = None, results_dir: Path | None = None) -> FastAPI:
  """Build a local explorer application with bounded in-memory results."""
  uploads: OrderedDict[str, tuple[bytes, dict[str, Any]]] = OrderedDict()
  runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
  lock = threading.Lock()
  executor: ThreadPoolExecutor | None = None
  storage = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
  latest_path = storage / "path-comparison-latest.json"

  def read_latest() -> dict[str, Any] | None:
    """Recover the latest completed comparison without starting any simulation."""
    try:
      record = json.loads(latest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      return None
    except (OSError, ValueError) as error:
      raise HTTPException(503, "The saved path comparison cannot be read.") from error
    if (
      not isinstance(record, dict)
      or record.get("status") != "complete"
      or not isinstance(record.get("result"), dict)
      or record["result"].get("kind") != "path_comparison"
      or not isinstance(record.get("id"), str)
    ):
      raise HTTPException(503, "The saved path comparison is invalid.")
    return record

  def save_latest(record: dict[str, Any]) -> None:
    """Atomically replace the previous completed comparison after success only."""
    storage.mkdir(parents=True, exist_ok=True)
    temporary = storage / f".path-comparison-{record['id']}.tmp"
    try:
      temporary.write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8"
      )
      temporary.replace(latest_path)
    finally:
      temporary.unlink(missing_ok=True)

  @asynccontextmanager
  async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run one simulation at a time and finish outstanding work on shutdown."""
    nonlocal executor
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="carfac-explorer")
    try:
      yield
    finally:
      executor.shutdown(wait=True, cancel_futures=True)
      executor = None

  app = FastAPI(title="CARFAC Explorer", lifespan=lifespan)

  @app.exception_handler(RequestValidationError)
  async def invalid_request(_request: Request, error: RequestValidationError) -> JSONResponse:
    """Keep invalid numeric inputs and exception contexts out of JSON errors."""
    details = [{key: entry[key] for key in ("loc", "msg", "type")} for entry in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})

  def submit(
    kind: str,
    operation: Callable[[Callable[[float, str], None]], dict[str, Any]],
  ) -> dict[str, str]:
    """Reserve the sole worker and retain the most recent four job records."""
    with lock:
      if any(job["status"] in {"queued", "running"} for job in runs.values()):
        raise HTTPException(409, "A simulation is already running; wait for it to finish.")
      if executor is None:
        raise HTTPException(503, "The simulation worker is unavailable.")
      identifier = uuid.uuid4().hex
      record: dict[str, Any] = {
        "id": identifier,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "message": "Waiting for the simulation worker.",
      }
      runs[identifier] = record
      while len(runs) > MAX_RESULTS:
        runs.popitem(last=False)

      def report(value: float, message: str) -> None:
        """Publish bounded progress without exposing a partially built result."""
        if not math.isfinite(value):
          return
        with lock:
          record.update(
            progress=max(record["progress"], min(1.0, max(0.0, value))), message=message
          )

      def execute() -> None:
        """Complete the immutable request snapshot or retain an explicit error."""
        with lock:
          record.update(status="running", message="Starting simulation.")
        try:
          result = operation(report)
          completed = {
            **record,
            "status": "complete",
            "progress": 1.0,
            "message": "Complete.",
            "result": result,
          }
          if kind == "path_comparison":
            save_latest(completed)
        except Exception as error:
          with lock:
            record.update(status="error", message="Simulation failed.", error=str(error))
        else:
          with lock:
            record.update(completed)

      executor.submit(execute)
    return {"id": identifier}

  @app.post("/api/runs", status_code=202)
  def start_run(request: RunRequest) -> dict[str, str]:
    """Validate and snapshot a tone or recording excerpt before simulation."""
    with lock:
      if any(job["status"] in {"queued", "running"} for job in runs.values()):
        raise HTTPException(409, "A simulation is already running; wait for it to finish.")
    audio_metadata = None
    try:
      if isinstance(request.stimulus, AudioStimulus):
        with lock:
          entry = uploads.get(request.stimulus.audio_id)
        if entry is None:
          raise HTTPException(404, "Recording not found; upload it again.")
        content, metadata = entry
        waveform = prepare_audio_excerpt(content, request.stimulus)
        audio_metadata = {**metadata, "sha256": hashlib.sha256(content).hexdigest()}
      else:
        waveform = prepare_stimulus(request)
    except ValueError as error:
      raise HTTPException(422, str(error)) from error

    def operation(progress: Callable[[float, str], None]) -> dict[str, Any]:
      """Run the already prepared stimulus rather than re-reading mutable uploads."""
      result = run_comparison(request, waveform=waveform, progress=progress)
      if audio_metadata is not None:
        result["input_audio"] = audio_metadata
      return result

    return submit("explore", operation)

  @app.post("/api/benchmarks", status_code=202)
  def start_benchmark(request: BenchmarkRequest) -> dict[str, str]:
    """Schedule a validated reference comparison on the same sole worker."""
    return submit("benchmark", lambda progress: run_benchmark(request, progress=progress))

  @app.post("/api/path-comparisons", status_code=202)
  def start_path_comparison(request: PathComparisonRequest) -> dict[str, str]:
    """Compare four paths from frozen sources with predeclared held-out bands."""
    return submit(
      "path_comparison", lambda progress: run_path_comparison(request, progress=progress)
    )

  @app.get("/api/path-comparisons/reference")
  def path_reference() -> dict[str, Any]:
    """Expose all clean observations, including the original MAT provenance."""
    return load_distortion_reference()

  @app.get("/api/path-comparisons/latest")
  def latest_path_comparison() -> dict[str, Any] | None:
    """Restore the last completed comparison after a browser or server restart."""
    return read_latest()

  @app.get("/api/runs/{identifier}")
  def get_run(identifier: str) -> dict[str, Any]:
    """Return a stable status snapshot and a result only after completion."""
    with lock:
      if identifier in runs:
        return dict(runs[identifier])
    saved = read_latest()
    if saved is not None and saved["id"] == identifier:
      return saved
    raise HTTPException(404, "Simulation result not found.")

  @app.get("/api/runs/{identifier}/export")
  def export_run(identifier: str) -> JSONResponse:
    """Download completed plot data and provenance without original audio bytes."""
    record = get_run(identifier)
    if record["status"] != "complete":
      raise HTTPException(409, "Only completed simulations can be exported.")
    return JSONResponse(
      record["result"],
      headers={"Content-Disposition": f'attachment; filename="carfac-{identifier}.json"'},
    )

  @app.post("/api/audio")
  async def upload_audio(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    """Retain at most four bounded recordings; decode excerpts only when run."""
    try:
      content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
      await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
      raise HTTPException(413, "Audio upload exceeds the 16-MiB limit.")
    try:
      metadata = _audio_metadata(content)
    except ValueError as error:
      raise HTTPException(422, str(error)) from error
    audio_id = uuid.uuid4().hex
    metadata.update(audio_id=audio_id, name=file.filename or "recording")
    with lock:
      uploads[audio_id] = (content, metadata)
      while len(uploads) > MAX_UPLOADS:
        uploads.popitem(last=False)
    return metadata

  @app.get("/api/health")
  def health() -> dict[str, str]:
    """Report whether the local server can accept requests."""
    return {"status": "ok"}

  @app.get("/api/reference")
  def reference() -> dict[str, Any]:
    """Return the packaged observed data with its provenance and limitations."""
    return load_reference()

  @app.get("/api/config")
  def config() -> dict[str, Any]:
    """Expose fixed experiment metadata and the validated default request."""
    return {
      "sample_rate": SAMPLE_RATE,
      "max_excerpt_s": 3,
      "max_upload_bytes": MAX_UPLOAD_BYTES,
      "max_upload_duration_s": MAX_UPLOAD_DURATION_S,
      "defaults": RunRequest().model_dump(),
      "benchmark_defaults": BenchmarkRequest().model_dump(),
      "path_comparison_defaults": PathComparisonRequest().model_dump(),
      "model": {
        "dtype": "float64",
        "agc_decimation": [1, 1, 1, 1],
        "state": "fresh silent initialization per condition",
        "output_units": "dB re 1 model unit (RMS)",
      },
    }

  directory = DEFAULT_FRONTEND_DIR if frontend_dir is None else frontend_dir
  if (directory / "index.html").is_file():
    app.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
  else:

    @app.get("/", response_class=HTMLResponse)
    def frontend_instructions() -> str:
      """Explain how to build the local UI when only the API is available."""
      return (
        "<h1>CARFAC Explorer API is running</h1>"
        "<p>Build the frontend, then restart this server:</p>"
        f"<pre>cd {html.escape(str(directory.parent))}\nnpm install\nnpm run build</pre>"
        '<p>API status: <a href="/api/health">/api/health</a></p>'
      )

  return app


@click.command()
@click.option(
  "--host",
  default="127.0.0.1",
  show_default=True,
  help="Bind address; keep the default for local research use.",
)
@click.option("--port", default=8768, type=click.IntRange(1, 65535), show_default=True)
@click.option(
  "--frontend-dir",
  type=click.Path(path_type=Path, file_okay=False),
  default=DEFAULT_FRONTEND_DIR,
  show_default=True,
)
def main(host: str, port: int, frontend_dir: Path) -> None:
  """Serve the local CARFAC explorer and its bounded simulation worker."""
  uvicorn.run(create_app(frontend_dir), host=host, port=port)


if __name__ == "__main__":
  main()
