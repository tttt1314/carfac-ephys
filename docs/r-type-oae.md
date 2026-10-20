# Jimmy's differentiable R-type OAE

This implementation adds the reflection path and synthetic-TEOAE fitting
experiment from Yong-Yue (Jimmy) Xu's 2026 thesis, §3.3–3.4.1. It uses the
repository's pinned official CARFAC and an explicit thesis preset. Existing
ABR/EFR model defaults and Ben's middle-ear options retain their behavior.

## Model

The preset uses 44.1 kHz, one ear, 69 channels from a 12 kHz first pole to a
100 Hz minimum pole, two-cap IHC, nonlinear CAR, delay buffers, closed-loop
multirate AGC with decimation `(8, 2, 2, 2)`, and `gc × 0.94`.

The return source is the AC-coupled BM output multiplied by per-channel
roughness `R`. Channel zero is basal. A source at channel `j` traverses `j`
one-sample inter-channel delays and all return filters from `j` through zero:

```text
acoustic pressure -> alpha -> forward middle ear -> CARFAC -> BM × R
   -> apical-to-basal first-order lowpass/delay chain -> beta
   -> return middle ear -> output pressure
```

Each return pole is `exp(-2*pi*CF*1.1/fs)`, with unit DC gain. Forward middle
ear: second-order 400–5000 Hz bandpass. Return middle ear: first-order
800–4000 Hz followed by second-order 400–5000 Hz. These are Jimmy's original
filter definitions; choosing the thesis preset applies them once.

`simulate()` accepts ear-canal pressure in Pa and returns a differentiable
one-dimensional JAX array in Pa. Every call starts at rest. For fitting parity,
the internal path retains the original pressure normalization and float32
operation order; the public output converts back to Pa.

```python
import jax
import jax.numpy as jnp
from carfac_ephys import r_type_oae

model = r_type_oae.build_model()
parameters = r_type_oae.initialize_parameters(model, health=0.75)
pressure_pa = jnp.zeros(1024).at[0].set(0.001)
run = jax.jit(r_type_oae.simulate)
oae_pa = run(model, parameters, pressure_pa)
gradients = jax.grad(lambda p: jnp.sum(run(model, p, pressure_pa) ** 2))(parameters)
```

## Prepare the archived inputs

Install with `uv sync --locked`. The six archived inputs remain outside Git in
`data/private/jimmy-teoae/`; no NAS access is required when the local research
archive is available:

```bash
uv run python scripts/prepare_teoae_data.py --research-dir /path/to/Research
```

The preparation script validates all twelve files against the checked-in
`jimmy_teoae_manifest.json` before copying them. Output folders use the labels
`Flat00`, `Slope20`, `Slope30`, `Notch4K`, `Notch3K`, and `Notch2K`.
The two slope inputs specifically come from the author's CARFAC archives
`Flat00_Slope20` and `Flat00_Slope30`. Their gain profiles match the separately
archived TDLM slope inputs, but their OAE waveforms differ. The manifest
preserves that distinction. The other four inputs come from the previously
audited six-pattern archive.

## Full fitting and evaluation

```bash
uv run carfac-ephys-fit-teoae \
  --data-dir data/private/jimmy-teoae \
  --output-dir output/teoae-thesis
```

The CLI runs all six patterns by default on CPU, using float32. It refuses to
overwrite an existing run. To request a subset, repeat `--profiles`, for
example `--profiles Flat00 --profiles Notch2K`. A short pipeline check uses
`--max-updates 2`; reduced budgets and subsets cannot pass the full six-pattern
reproduction verdict.

Each pattern fits the 80 dB positive-polarity click, row **index 2** of the
archived `(6, 2205)` array. The raw target slice is `[132:926]`, normalized by
20 µPa. The model click has five nonzero samples preceded by 128 samples;
those preceding samples are removed before starting the zero-state return
path, and its fitting window is `[88:882]`.

Trainable parameters are per-channel sigmoid-constrained OHC health `H`,
roughness `R`, input gain `alpha`, return gain `beta`, velocity scale, and
velocity offset. CAR gain coefficients `ga`, `gb`, `gc`, and AC coupling
remain fixed. The loss follows the author's exact definitions:

| Term | Fixed weight, Eq. (3.16) |
|---|---:|
| Multi-resolution STFT | 5.9165 |
| Time RMSE in normalized units | 0.5683 |
| Log10 FFT magnitude MSE | 454.0997 |
| Squared adjacent-health differences, summed | 188.4442 |
| Sorted-roughness Gaussian quantile MSE | 79.6671 |

STFT FFT/window sizes are `(512, 256, 128)`, with hops `(256, 128, 30)`.
Each resolution adds spectral convergence and natural-log magnitude error;
the final STFT term averages the three resolutions. Spectrum loss uses the
author's truncated frequency-bin endpoints from 600 to 8000 Hz.

Each pattern starts independently at health `0.99`, `0.75`, and `0.5`, with
seed-42 JAX float32 normal roughness, `alpha=0 dB`, and `beta=-50 dB`. Adam
uses health learning rate `0.05`; auxiliary learning rate follows the
1500-step cosine schedule from `0.002` to `0.001`. The original inclusive
loop makes at most **1501 updates**. Early stopping tests pre-update loss
`<0.5` after applying that update. The winner minimizes the last pre-update
loss; its updated parameters and recomputed final loss are saved separately.
Nonfinite trajectories fail explicitly. No Optuna search is performed.

Audiogram labels are absent from the `fit()` interface. Only after start
selection, `evaluate()` probes every model CF with a 10 dB SPL pure tone for
100 ms and takes the last 50 ms RMS of the corresponding channel. Probe
signals directly enter CARFAC, without input gain or middle-ear filtering.
The original healthy model supplies the baseline; fitted `H`, velocity scale,
and offset supply the impaired response. The original negative-gain floor,
CF interpolation, and two nearest-neighbor target mappings are retained.
These values describe **gain reduction**, not clinical hearing thresholds.

## Artifacts and acceptance

The output includes `protocol.json` with input hashes, source provenance and
runtime versions; a folder per pattern with all three parameter trees and
loss histories; selected parameters; evaluation arrays; and TEOAE waveform
and spectrum plots. `report.md`, `report.json`, and `gain-comparison.png`
collect the Table 3.1 comparison and the six gain curves.

Every pattern must meet all three predeclared upper limits:

| Pattern | Audiogram MAE (dB) | Time RMSE (mPa) | Spectrum MAE (dB) |
|---|---:|---:|---:|
| Flat00 | 1.40 | 0.662 | 3.68 |
| Slope20 | 2.42 | 0.381 | 1.88 |
| Slope30 | 3.67 | 0.322 | 1.97 |
| Notch4K | 5.25 | 0.534 | 2.90 |
| Notch3K | 2.65 | 0.338 | 4.26 |
| Notch2K | 5.20 | 0.317 | 3.49 |

These equal Table 3.1 plus 1 dB, 0.1 mPa, and 1 dB, respectively. Lower errors
also pass. An unsuccessful full run exits nonzero after saving its report.
The historical thesis fitted checkpoints are unavailable, so the comparison
tests fresh-fit performance rather than identical historical parameter values.
Input aliases, weights, and thresholds remain fixed when a case fails.

## Tests and local benchmarks

The regular suite uses synthetic inputs with readable expectations:

- A three-channel impulse crosses three lowpasses and two sample delays.
- RMS pressure amplitude obeys `A(beta + 20) = 10*A(beta)` and
  `dA/dbeta = A*ln(10)/20`.
- Simple linear losses verify stopping after the update and choosing the
  winner by its last pre-update loss. These tests use SGD to make the
  arithmetic explicit; they test the fitting loop, not Adam convergence.

The suite also checks the target window, acceptance limits, corrupted inputs,
and incomplete-run verdicts. Run it with:

```bash
uv run pytest tests/test_r_type_oae.py tests/test_teoae_fitting.py tests/test_teoae_cli.py
```

Comparisons against Jimmy's original implementation and the full six-pattern
training are separate local benchmarks. Their fixtures, comparison tools,
checkpoints, logs, and plots stay in Git-ignored local directories. Ordinary
tests and CI do not require the author's source archive or private inputs.

Record benchmark results in the pull request: the runtime versions, input and
implementation hashes, original-code comparison tolerance, and all six
patterns' three metrics and pass/fail verdicts. The fitting CLI above records
the experiment under `output/`; its generated report supplies the result
table. Distinguish these local results from the checks run by CI.
