"""Copies the six audited archives into ignored local training inputs."""

import argparse
import json
import shutil
from pathlib import Path

from carfac_ephys.teoae_cli import MANIFEST_PATH, file_sha256


def main() -> None:
  """Validates all source hashes before copying any training input."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--research-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, default=Path("data/private/jimmy-teoae"))
  args = parser.parse_args()
  manifest = json.loads(MANIFEST_PATH.read_text())
  copies = []
  for profile, sources in manifest["profiles"].items():
    for kind, filename in (("oae", "oae.npy"), ("profile", "profile.txt")):
      source = args.research_dir / sources[kind]["source"]
      if file_sha256(source) != sources[kind]["sha256"]:
        raise ValueError(f"Source archive hash mismatch: {source}")
      copies.append((source, args.output_dir / profile / filename))
  for source, destination in copies:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
  shutil.copyfile(MANIFEST_PATH, args.output_dir / "manifest.json")
  print(f"Prepared {len(copies)} verified files in {args.output_dir}")


if __name__ == "__main__":
  main()
