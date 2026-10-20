# Differentiable R-type OAE and TEOAE fitting

This implementation adds the reflection path and synthetic-TEOAE fitting
recipe from Yong-Yue (Jimmy) Xu's 2026 thesis, §3.3–3.4.1. It uses the
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

## Fit supplied click responses

Install with `uv sync --locked`. Put each input in a named folder under
`--data-dir`; folder names are chosen by the caller:

```text
data/private/teoae/
  my-click/
    oae.npy
    profile.txt  # optional gain-reduction labels for post-fit evaluation
```

`oae.npy` contains one finite 794-sample pressure waveform in Pa, aligned to
the model's fitting window `[88:882]` after the click. Optional `profile.txt`
contains two rows: frequency in Hz and
gain reduction in dB. Labels are loaded only after fitting and start selection.

```bash
uv run carfac-ephys-fit-teoae \
  --data-dir data/private/teoae \
  --output-dir output/teoae-fitting
```

The CLI discovers all folders containing `oae.npy`. Repeat `--profiles` to
select particular folder names. It records the actual input SHA-256 hashes;
no author-specific data manifest is required. It runs on CPU using float32
and refuses to overwrite an existing run. Use `--max-updates 1` for a short
pipeline check.

Each input fits an 80 dB positive-polarity click response, normalized by
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

## Saved results

`protocol.json` records the model configuration, training settings, runtime
versions, and input and implementation hashes. Each input gets a folder with
all three fitted parameter trees and loss histories, selected parameters,
and `result.json`. With gain labels, it also contains `evaluation.npz` and
all three error metrics. The top-level `report.json` collects these results.

Paper-specific reference scores, acceptance thresholds, plots, and the
six-pattern reproduction driver are local benchmark tooling. They do not
control the public CLI's results or exit status.

## Tests and local benchmarks

The regular suite uses synthetic inputs with readable expectations:

- A three-channel impulse crosses three lowpasses and two sample delays.
- RMS pressure amplitude obeys `A(beta + 20) = 10*A(beta)` and
  `dA/dbeta = A*ln(10)/20`.
- Simple linear losses verify stopping after the update and choosing the
  winner by its last pre-update loss. These tests use SGD to make the
  arithmetic explicit; they test the fitting loop, not Adam convergence.

The suite also checks pressure normalization, input selection and hashes, and a
one-update CLI run with synthetic data and no gain labels. Run it with:

```bash
uv run pytest tests/test_r_type_oae.py tests/test_teoae_fitting.py tests/test_teoae_cli.py
```

Comparisons against Jimmy's original implementation and the full six-pattern
training are separate local benchmarks. Their fixtures, comparison tools,
checkpoints, logs, and plots stay in Git-ignored local directories. Ordinary
tests and CI do not require the author's source archive or private inputs.

Record local benchmark results in the pull request, together with the data
and runtime details. Keep the original-code comparisons and six-pattern
experiment separate from the ordinary checks run by CI.
