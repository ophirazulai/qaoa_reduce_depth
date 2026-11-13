# QAQA reduce depth

Search a finite grid of transpiler seeds and SABRE wait-time weights, including
a baseline with wait time disabled. For each input circuit, save the transpilation
with the smallest two-qubit depth (D2) and report its seed, weight, and two-qubit
gate count (N2). Run locally with multiple processes or submit parallel LSF jobs.

## Install

Use Python 3.12 or newer, Git, a C/C++ build toolchain, and a recent stable Rust
toolchain (`rustc` and `cargo` on `PATH`). Building the Qiskit fork can take several
minutes and needs network access to GitHub, PyPI, and Rust package registries.
`uv` is recommended; an ordinary virtual environment and pip also work.

```bash
cd /u/ophirazu/projects/qaqa_reduce_depth
uv venv --python 3.12
source .venv/bin/activate
CARGO_BUILD_JOBS=4 uv pip install -e '.[dev]'
qaqa-reduce-depth --help
```

Without uv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CARGO_BUILD_JOBS=4 python -m pip install -e '.[dev]'
```

`pyproject.toml` installs Qiskit from
[`AI-native-Systems-Research/qiskit-evolved`, branch `sabre-wait-time-aware-swap-scoring`](https://github.com/AI-native-Systems-Research/qiskit-evolved/tree/sabre-wait-time-aware-swap-scoring).
It pins commit `9dedcf6f64a66344cb31364e846f71d020adcf9d`, the branch revision
used to develop and test this project. This is a source dependency, not the PyPI
Qiskit release. The fake backend provider is pinned to
`qiskit-ibm-runtime==0.52.1`. No IBM account or hardware access is required.

The program checks for the fork's wait-time and SABRE layout APIs before starting
a sweep. If that check fails, reinstall this project in the active environment.
Do not replace its Qiskit dependency with `pip install --upgrade qiskit`.

## Input circuits

Place QPY files under `./circuits`, or select another directory with `--circuits`.
The initial data is copied from
`/u/ophirazu/projects/transpilation_compare/circuits_subset` without modifying
the source. Circuit artifacts are excluded from Git.

The program recursively discovers regular `.qpy` files in sorted order. It does
not follow directory symlinks and rejects QPY file symlinks. Each circuit in a
multi-circuit QPY file is an independent input; reports include its zero-based
index. Identical basenames in different directories cannot collide in output.

Inputs are copied into the run directory and checksummed before workers start.
Workers read those snapshots, so subsequent edits to the originals do not alter
the sweep. Empty/malformed QPY files, circuits wider than the target, and dynamic
control flow fail preparation. Static circuits may include measurements,
barriers, and unbound parameters; unbound parameters remain symbolic.
A failed preparation names the offending file and removes what it created, so
the same output directory can be reused after fixing the input.

## Run locally

```bash
source .venv/bin/activate
qaqa-reduce-depth run
```

Defaults:

| Setting | Value |
| --- | --- |
| Input directory | `./circuits` |
| Output directory | `./output` |
| Target | `qiskit_ibm_runtime.fake_provider:FakeBoston` |
| Seed count | 30, seeds `0..29` |
| Weights | `0.00001 0.00005 0.0001 0.00025 0.0005 0.001` |
| Baseline | Always included, once per seed |
| Optimization level | 3 |
| Local processes | 5 |
| Transpilations per batch/job | 5 |
| Internal SABRE layout and swap trials | 20 each |
| Native threads per worker | 1 |

For `C` input circuits, `S` seeds, and `W` distinct weights, there are exactly
`C × S × (W + 1)` trials. Defaults give **210 trials per input circuit**. Each
trial always starts from the original input, not a previously transpiled output.

Customize the search:

```bash
qaqa-reduce-depth run \
  --circuits ./circuits --output ./output-custom \
  --seeds 50 --seed-start 100 \
  --weights 0.00001 0.00005 0.0001 0.00025 0.0005 0.001 \
  --workers 5 --optimization-level 3 --sabre-trials 20
```

Weights must be finite and nonnegative; duplicates are removed and values sorted.
Weight `0` is allowed, but is distinct from baseline: it uses the wait-time
preset with zero weight and no decay. Counts must be positive, and all seeds
must fit unsigned 32-bit integers. Output must be a new or empty directory,
separate from the input directory. Existing runs are never overwritten.

Use `python -m qaqa_reduce_depth` in place of the console command if preferred.

## Choose a target

`FakeBoston`, `FakeBoston()`, and the full import path select the default backend:

```bash
qaqa-reduce-depth run --target FakeBoston --output ./output-boston
qaqa-reduce-depth run \
  --target qiskit_ibm_runtime.fake_provider:FakeBrisbane \
  --output ./output-brisbane
```

Other targets use `module:factory`. The factory must be importable in every
worker environment and take no arguments, returning a Qiskit BackendV2. Supply
a factory function in an installed module for a custom backend. Keep its target
configuration deterministic across workers; Python expressions are not evaluated.

## Run with LSF

Install this project in a shared filesystem environment that all compute nodes
can access at the same absolute path. The repository, virtual environment, output
directory, and its input snapshots must be shared. `bsub` must be available and
your shell must have the site's LSF environment loaded.

```bash
source .venv/bin/activate
qaqa-reduce-depth run \
  --executor lsf --transpilations-per-job 5 \
  --lsf-queue normal --lsf-wall-minutes 120 \
  --lsf-memory-gb 10 --lsf-memory-unit GB \
  --output ./output-lsf
```

The number of submitted jobs is `ceil(total trials / transpilations-per-job)`.
Each job requests one CPU and runs at most `--transpilations-per-job` trials
sequentially. One trial is one input circuit, one seed, and one baseline/weight
setting. The last job may be smaller. With the default grid and batch size,
there are **42 jobs per input circuit**. All jobs are submitted without a
concurrency cap; LSF controls when they run. `--workers` applies only to local
execution, which uses the same batches and winner selection.

The coordinator starts one `bsub -K` client per job to submit it and wait for
completion. It stays running until all jobs finish and then builds
the report. Keep that process alive using your site's usual persistent session
or controller-job workflow; this command does not detach a separate controller.
There is no automatic resubmission or resume scheduler. If interrupted, accepted
jobs may continue; inspect `bjobs` and `_work/shards/*/bsub.log` before deciding to
cancel them with your site's `bkill` command. After workers finish, `collect`
can rebuild reports from completed results.

LSF-specific options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--lsf-queue` | `normal` | Queue; pass an empty string for site default |
| `--lsf-project` | unset | Optional `bsub -P` project |
| `--lsf-wall-minutes` | `120` | Whole-job wall limit in minutes |
| `--lsf-memory-gb` | `10` | Whole-job memory request, in GiB |
| `--lsf-memory-unit` | `GB` | Site units for both `-M` and `rusage[mem=]` |
| `--bsub` | `bsub` | Submission executable name or path |

The `GB` default matches the source project's CCC site configuration. Use `MB`
or `KB` if your site's limits and reservations use those units. Allow enough
wall time and memory for the entire batch, backend, and QPY loading. Local mode
does not impose per-trial memory or wall-time limits.

## How the heuristic is enabled

Both layout and routing are explicitly set to SABRE. The code traverses nested
pass-manager flow controllers and configures **every `SabreLayout` and
`SabreSwap`**, including the routing performed internally by layout.

Baseline uses the branch's decay heuristic: basic weight `1`, lookahead weight
`0.5 / target_width`, and decay `(0.001, 5)`, with no wait-time component.
Weighted trials use the branch's wait-time preset: the same basic and lookahead
terms, `.with_wait_time(weight)`, and no decay term. This compares the two
heuristic modes; it is not an experiment holding decay enabled in both arms.
All passes use an attempt limit of `10 × target_width`.

SABRE's internal trial counts are fixed by `--sabre-trials`; they do not depend
on available CPU cores. The default of 20 matches the optimization-level-3 search
budget used in `transpilation_compare`'s `subset_weight_sweep_50_seeds` experiment.
Reducing this value searches fewer candidates to save time; increasing the number
of external seeds does not reproduce the same internal search. Each external seed
is passed as `seed_transpiler`.
Native thread counts are fixed to one before Qiskit imports, preventing nested
parallelism from multiplying the requested workers. The requested seed grid is
separate from the internal SABRE trials.

## Metrics, selection, and outputs

**N2** counts two-qubit gates in the transpiled circuit, excluding directives
such as barriers. **D2** is the longest dependency path when two-qubit gates
cost one and other operations cost zero. Zero-cost operations still preserve
wire dependencies; a barrier can synchronize otherwise independent paths.
After target translation these are native two-qubit gates. D2 is a layer count,
not calibrated execution duration. Dynamic control flow and unexpected output
gates with more than two qubits are rejected rather than assigned approximate
metrics.

The winner minimizes `(D2, N2, baseline-first, weight, seed)`: first D2, then N2,
then baseline over a weight, then the smaller weight and seed. This provides
deterministic tie handling independent of completion order. The result is the
minimum among the requested trials, not a claim of a globally optimal circuit.

For each complete input, the report also compares the best wait-time D2 across
all requested seeds and weights with the best baseline D2 across all requested
seeds. Weight `0`, if requested, belongs to the wait-time group. The **D2 ratio**
is `best wait-time D2 / best baseline D2`: below `1` means wait time improves
depth, `1` means equal depth, and above `1` means worse depth. The **N2 ratio**
is `wait-time N2 / baseline N2` for those same best trials, selected by D2 and
then N2. Below `1` means fewer two-qubit gates with wait time. Both groups' D2
and N2 values and both ratios are included in the report and `results.json`.
Incomplete inputs have no comparison. A zero baseline value makes its ratio
undefined, shown as `N/A` in the report and `null` in `results.json`.

```text
output/
  report.md             winners, seeds, weights, D2, N2, and baseline D2/N2 ratios
  results.json          structured per-input selection and failures
  trials.json           every completed trial's metrics, timing, or error
  plan.json             search grid, versions, input identities and checksums
  execution.json        execution/submission failures
  best/
    e20-c20-z3__base__p1.qpy  selected circuit, named after its input
  _work/
    inputs/             immutable input snapshots
    shards/000000/      batch results and best candidate per input in that batch
                        plus job.sh, submission.json, and logs in LSF mode
```

Files in `best/` preserve their input filenames and relative subdirectories.
For multi-circuit inputs, an index is added: `bundle.qpy` produces
`bundle__circuit_0.qpy`, `bundle__circuit_1.qpy`, and so on. An additional numeric
suffix is added if an indexed name conflicts with an input file or directory.
The report and `results.json` link to the exact output paths.

Each selected QPY file contains one circuit, including its physical layout.
QPY candidates are saved as the trials run, so collection copies the winning
artifact instead of recompiling it. Only batch winners are retained, limiting
temporary disk usage compared with keeping every transpilation. `_work` is kept
to support verification and report regeneration; reserve disk for snapshots
and these intermediate candidates.

An input receives a published winner only if **all** its planned trials succeed.
Missing jobs and transpilation errors mark it incomplete; other fully completed
inputs can still have winners. Exit status is `0` for a complete sweep, `1` for
failure/incomplete results, `2` for CLI argument errors, and `130` on interruption.
Inspect `report.md`, `results.json`, and LSF logs for details. Use a new output
directory to rerun a failed search.

Ctrl-C or `kill` (SIGINT or SIGTERM) on the coordinator stops a local sweep's
workers immediately. Batches that already finished are kept, and `collect`
reports the remaining ones as missing. For LSF, see the note on accepted jobs above.

Regenerate reports without transpilation:

```bash
qaqa-reduce-depth collect --output ./output-lsf
```

Collection validates batch identities and candidate checksums. Do not edit or
remove `_work` if you need to run `collect` again.

## Tests

```bash
python -m pytest -q
ruff check src tests
```

Tests create synthetic circuits. They verify both SABRE passes, equivalence up
to the final layout, deterministic seeds, dependency-depth measurement, QPY
round trips, exhaustive grid selection, multiple circuits per file, parallel
local execution, and LSF execution through a fake submission executable.
The test suite never submits real cluster jobs or reads the supplied circuit
collection.
