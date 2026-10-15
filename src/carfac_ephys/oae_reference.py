"""Load the attributed, observed DPOAE comparison used by the explorer."""

import json
from importlib.resources import files
from typing import Any


def load_reference() -> dict[str, Any]:
  """Return the single-ear Dewey distortion-component reference and its limits.

  The stored amplitudes and phases are published observations, with primary
  frequencies inferred from the published ratio. They are not model outputs.
  A fresh object is returned so callers can annotate a comparison safely.
  """
  path = files("carfac_ephys").joinpath("data", "dewey_dpoae_reference.json")
  return json.loads(path.read_text(encoding="utf-8"))
