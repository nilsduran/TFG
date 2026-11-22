# Usage guide

Detailed reference for installing, running and extending the `br_analysis` pipeline.
For an overview of the project and its findings, see the [main README](../README.md).

## Installation

Python 3.12 is the supported runtime. Dependency ranges in `pyproject.toml`
are tested against the current Python 3.12 environment and deliberately exclude
Python 3.13, where the local MNE/Numba stack has shown import problems.

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[deep,explain,dev]"
```

### Project configuration

`pyproject.toml` is the standard Python project configuration file. It is not
executed manually: `pip`, setuptools, pytest and Ruff read it automatically.
In this repository it defines:

- the base scientific dependencies;
- the optional `deep`, `explain` and `dev` dependency groups;
- the editable `src/br_analysis` package layout;
- the `br-analysis` command-line entry point;
- the pytest and Ruff settings.

Install only what an environment needs:

```text
python -m pip install -e .                 # classical models and CLI
python -m pip install -e ".[deep]"         # add PyTorch models
python -m pip install -e ".[explain]"      # add SHAP
python -m pip install -e ".[deep,explain,dev]"  # full development setup
```

The `-e` editable installation points Python at the local source tree, so code
changes are available immediately. Rerun the installation command only after
changing dependencies or creating a new environment. `requirements.txt` is a
short convenience installer for `-e .[deep,explain]`; `pyproject.toml` remains
the single source of dependency and packaging metadata. Dependencies use
compatible lower and upper bounds rather than exact pins: this prevents known
breaking upgrades while still allowing bug-fix releases.

Place preprocessed FIF epoch files directly in:

```text
data/
  P002.fif
  P003.fif
  ...
```

The default output directory is `artifacts/`. Alternative local locations can
be provided without changing code:

```text
BR_DATA_DIR=D:\eeg\epochs
BR_ARTIFACTS_DIR=D:\eeg\results
```

The package contains no platform detection, mounted-drive logic or absolute
paths. A remote runtime should install the same package and provide these same
directories externally.

## Tasks

Tasks are declarative combinations of event codes:

| Task | Classes |
| --- | --- |
| `c_vs_ip` | C `[1, 2]` vs IP `[3, 4, 5, 6]` |
| `im_vs_ip` | IM `[7, 8]` vs IP `[3, 4, 5, 6]` |
| `c_vs_ip_vs_im` | C vs IP vs IM |
| `ipr_vs_ipg` | IPR `[3, 4]` vs IPG `[5, 6]` |
| `cr_vs_cg` | CR `[1]` vs CG `[2]` |
| `r_vs_g` | red `[1, 3, 4]` vs green `[2, 5, 6]` |

Additional binary or multiclass tasks can be added in
`src/br_analysis/tasks.py` without creating another loader or notebook.

## Methodological contract

- Every FIF that passes task-independent quality control is included.
- Subjects are never selected from effect size or decoding performance.
- Trials are never undersampled before splitting.
- Class imbalance is handled using training-fold weights or priors.
- The latest trials remain available for chronological holdout.
- Primary metrics are balanced accuracy and macro F1.
- All models store class-aligned probability matrices.
- Learned normalization is fitted using training data only.

### Development and final evaluation

For every subject, the first 80% of chronological trials is the development
period and the final 20% is an untouched test set. Classical hyperparameters
are selected only within development data using temporal-block
`StratifiedGroupKFold`; the selected model is fitted on the complete 80% and
evaluated once on the final holdout. A subject is skipped when development or
holdout does not contain every task class.

Deep models use a chronological validation subset inside the development
period to select the number of epochs by macro F1. They are then reinitialized
and fitted on the full development period for that number of epochs before
the single final evaluation. Training still minimizes weighted
cross-entropy: macro F1 is a non-differentiable selection metric, not a neural
loss.

## Preprocessing

All models start from one canonical MNE loader. It reconstructs event IDs,
preserves trial order and caches each subject before task selection. Tasks,
channel subsets and shorter analysis windows are then derived without reading
or filtering the FIF files again.

Five presets cover the maintained models and scientific extensions:

| Preset | Processing | Main consumers |
| --- | --- | --- |
| `broadband` | 1-40 Hz, 128 Hz | Riemann, SVC Agnostic, EEGNet |
| `band_limited` | 4-13 Hz, 128 Hz | LR, clustering, ST-GNN |
| `pretrained` | 1-40 Hz, 128 Hz | ST-EEGFormer |
| `labram_pretrained` | 1-40 Hz, 200 Hz | LaBraM |
| `erp` | 0.1-30 Hz, 250 Hz, -740 to 1500 ms | Temporal onset |

Model-specific covariance, envelope, DE graph and pretrained tensor
representations are derived after this shared preprocessing.

## Models

The maintained benchmark contains:

1. `lr_theta`
2. `lr_alpha`
3. `riemann`
4. `svc_agnostic`
5. `svc_clustering`
6. `eegnet_agnostic`
7. `eegnet_clustering`
8. `st_gnn`
9. `st_eegformer`
10. `labram`

The previous experiment-guided configuration has been removed. Cluster-based
models are extensions; agnostic models remain the primary comparison.
ST-EEGFormer and LaBraM download revision-pinned official checkpoints from
Hugging Face and never run from random initialization. Downloads are retained
in the normal Hugging Face cache and reused across subjects and future runs.

## Commands

List configured tasks:

```text
br-analysis help
br-analysis list-tasks
br-analysis list-models
```

The help output also includes a runnable example and a coverage matrix with
models as rows and tasks as columns. `X` marks completed combinations found in
`artifacts/`.

Validate local FIF metadata:

```text
br-analysis validate-data
```

Prepare every preprocessing variant used by the configured models:

```text
br-analysis prepare-data
```

This performs the expensive filtering and resampling once for all event codes.
It includes the dedicated temporal-onset ERP variant and also caches
Hanning-window theta/alpha power, corrected against the
`-740` to `-240 ms` baseline, for the clustering analysis. Every decoding task
then derives its classes from the same shared caches. Use `--force` to rebuild
them, `--skip-power` to omit power, or `--models lr_theta riemann` to prepare
only the signal variants needed by selected models.

Run one classical model:

```text
br-analysis run --task c_vs_ip --model riemann
br-analysis run --task r_vs_g --model svc_agnostic
```

Use `all` on either axis to run a batch:

```text
br-analysis run --task all --model riemann
br-analysis run --task c_vs_ip --model all
br-analysis run --task all --model all
```

Without `--force`, a batch skips every task/model combination that already has
complete artifacts, even if the implementation signature has changed. This
makes `all` a way to fill missing cells rather than retrain the coverage
matrix. The batch continues after an unavailable combination and prints a
final completed/skipped/failed summary.

Run a deep model:

```text
br-analysis run --task c_vs_ip --model eegnet_agnostic
br-analysis run --task c_vs_ip --model st_eegformer
br-analysis run --task c_vs_ip --model labram
```

`im_vs_ip` is a prespecified exploratory exception to the classical-evidence
gate. Deep models are allowed to run for this task even when its classical
group mean is at chance; the report labels that interpretation explicitly.
This exception does not turn a null classical result into positive evidence.

### Temporal decoding onset

Estimate when pointwise ERP patterns first provide decodable information:

```text
br-analysis temporal-onset --task all --jobs 4
```

The dedicated ERP cache covers `-740` to `1500 ms`, uses a `0.1-30 Hz`
bandpass at 250 Hz, and subtracts each trial's `-740` to `-240 ms` baseline.
The analysis begins at the start of the stimulus ramp (`-240 ms`) and also
marks full stimulus onset (`0 ms`). Train-only standardisation and an
OAS-shrunk LDA equivalent to sklearn's `lsqr` formulation are fitted
separately at every 4 ms sample using the scalp voltage pattern.

The first chronological 80% supplies blocked out-of-fold predictions and
discovers the first max-T FWER-corrected run lasting at least 20 ms. Only that
preselected 20 ms interval is then tested once in the untouched final 20%.
Holm correction is applied across all six tasks when compatible artifacts for
the complete task family exist. BY-FDR and six-fold averaged pseudo-trial
decoding repeated 100 times are sensitivity analyses, not the primary
inferential criterion.

Individual onsets require their own block-aware label permutations and
max-T correction. Subjects without a detected onset are censored at 1500 ms
rather than discarded. A hierarchical subject-and-trial bootstrap reports an
onset interval and the percentage of bootstrap samples in which an onset was
detected. Numerical performance above chance alone is never labelled
significant.

Outputs under `artifacts/temporal_onset/` include the group and subject curves,
out-of-fold predictions, pseudo-trial sensitivity, independent confirmation,
pre-stimulus and channel controls, bootstrap distribution and the common
report figure. Because the available inputs are epoched FIF files rather than
continuous recordings, the recorded source filters and this onset limitation
are included in the metadata.

### Inter-subject transfer

Evaluate one compact transfer-learning family instead of repeating every model:

```text
br-analysis inter-subject --task all
```

Each target subject is fully excluded from model training. The baseline pools
the remaining subjects with a Log-Euclidean MDM classifier. The adapted variant
first applies Euclidean Alignment independently to every source subject and
uses only the first 80% of the target subject's trials, without labels, to
estimate its covariance reference. The final 20% remains untouched. No subject
identity classifier is trained or needed. Results and paired alignment gains
are stored under `artifacts/inter_subject/` and embedded in the common report.

This follows the low-cost alignment literature before considering another deep
model family: [Riemannian Procrustes Analysis](https://pubmed.ncbi.nlm.nih.gov/30596565/)
and [Euclidean Alignment](https://arxiv.org/abs/1808.05464). EEGNet adaptation
is a possible second stage if aligned Riemannian transfer is promising.

For an offline run, `--checkpoint MODEL=DIRECTORY` may point to a complete
Hugging Face snapshot containing both `config.json` and its weights. A
standalone `.safetensors` file is intentionally rejected because it does not
identify the architecture or preprocessing contract.

### Kaggle GPU from VS Code

Kaggle is the supported remote terminal workflow for transformer training.
Colab's free hosted runtime does not expose an equivalent job-submission CLI;
its documented local-runtime connection uses the computer running Jupyter.

Install the optional official Kaggle CLI and authenticate outside the
repository:

```powershell
python -m pip install -e ".[cloud]"
kaggle auth login
```

Alternatively, generate a token under Kaggle account settings and set it only
for the current PowerShell session:

```powershell
$env:KAGGLE_API_TOKEN = "your-token"
```

Do not place the token in `.env`, notebooks, metadata or Git. Publish two
private datasets once, replacing `USER` with the Kaggle username:

```powershell
br-analysis kaggle publish-source --ref USER/br-analysis-source
br-analysis kaggle publish-data --ref USER/binocular-rivalry-eeg
```

`publish-data` does not upload the 18 GB of FIF files. It exports the shared
preprocessing variants needed by the two transformers, including ST-EEGFormer's
128 Hz tensors and LaBraM's 200 Hz tensors. These files retain unaggregated
epochs rather than the original FIF containers. The staging directory uses hardlinks under
`artifacts/_cloud/kaggle/`, so it does not duplicate even that cache on disk.
Run `br-analysis prepare-data --skip-power` first if a current cache is
missing. After preprocessing changes, publish a new dataset version with
`--update`. Official model weights are downloaded directly from Hugging Face
by the internet-enabled private kernel and cached for the duration of the job.
Pass `--models` to publish another subset of torch preprocessing variants.

Submit a private GPU kernel directly from the VS Code terminal:

```powershell
br-analysis kaggle submit `
  --kernel USER/br-st-eegformer-c-vs-ip `
  --source-dataset USER/br-analysis-source `
  --data-dataset USER/binocular-rivalry-eeg `
  --task c_vs_ip `
  --model st_eegformer
```

Use a different kernel slug for LaBraM. Follow and recover the run with:

```powershell
br-analysis kaggle status --kernel USER/br-st-eegformer-c-vs-ip
br-analysis kaggle download --kernel USER/br-st-eegformer-c-vs-ip
br-analysis report
```

Downloaded results are merged into the normal local `artifacts/` hierarchy and
therefore appear in the common report. The download refuses to overwrite an
existing task/model run unless `--replace` is given explicitly.

Add `--explain` to a model run to generate SHAP values from development-only
backgrounds and final-holdout samples. Deep-learning and SHAP runs are skipped
when at least two completed classical baselines indicate that the task is
effectively at chance. Use `--allow-unpromising` to override this guard
deliberately.

The CLI is the recommended way to run long experiments reproducibly. Model
notebooks call the same Python runners and are intended for interactive
inspection or development, so using either route produces the same artifact
layout.

Interactive runs display `tqdm` progress with percentage and ETA. Batch
commands track completed task/model combinations, every model tracks subjects,
and deep models also show F1-based epoch selection and the final fit. Progress bars disable themselves in
non-interactive logs and tests.

Classical models can train independent subjects concurrently:

```text
br-analysis run --task c_vs_ip --model riemann --jobs 4
br-analysis run --task all --model svc_agnostic --jobs 4
```

Workers share the dataset in memory and each subject still runs its folds
sequentially. Numerical-library threads are limited inside the worker pool to
avoid CPU oversubscription. `--jobs` intentionally does not parallelize deep
models: simultaneous training jobs on one GPU usually increase memory use and
reduce reproducibility without improving throughput.

Compatible completed runs are reused automatically. Pass `--force` only when
you intentionally want to retrain:

```text
br-analysis run --task c_vs_ip --model riemann --force
```

There is exactly one saved run per task/model combination. Recomputing with
`--force` replaces that combination's artifacts; there are no run IDs or
`latest` subdirectories.

## Clustering

The clustering analysis starts from one shared broadband epoch cache covering
`-740` to `1500 ms`. A second cache stores 500 ms Hanning-window power in dB
relative to the `-740` to `-240 ms` baseline. It contains theta and alpha as
separate axes; clustering is always run independently for each band.

Two outputs deliberately remain distinct:

- the article ROI analysis uses Fz/FCz/Cz for theta and posterior-occipital
  channels for alpha, with temporal permutation clusters and FWER p-values;
- the reduction analysis finds one connected whole-head space-time region per
  band, with minimum size constraints, whether or not it is significant.

Condition means are calculated within each subject before group tests, so
trials are not treated as independent subjects. The notebook reports
subject-to-group Jaccard divergence and saves one `theta.npz` and `alpha.npz`
result per task under `artifacts/clustering/`.

The candidate masks are not yet injected into cross-validation. Doing so
directly would leak task labels. Once accepted visually, they must be defined
from an independent contrast/cohort or fitted inside each training fold before
being used by `svc_clustering` and `eegnet_clustering`.

## Subject identification

The subject-identification control uses chronological per-subject holdout and
an OAS covariance, Riemannian tangent-space and balanced logistic-regression
pipeline. It measures identity leakage without reusing the decoding labels as
the prediction target.

## Explainability

SHAP is the primary explainability interface:

- permutation SHAP for tabular classical representations;
- tangent-space SHAP for the Riemannian pipeline;
- Gradient SHAP for PyTorch models;
- stratified backgrounds sampled only from training folds;
- explanations computed only for held-out samples;
- class-specific values for binary and multiclass tasks.

Attribution magnitudes are aggregated within each model. Raw SHAP values from
different input spaces are not treated as directly comparable. Integrated
Gradients is no longer part of the default workflow.

## Notebooks

The notebooks are narrative interfaces over the Python package:

| Notebook | Purpose |
| --- | --- |
| `01_eda.ipynb` | descriptive class and waveform analysis |
| `02_clustering.ipynb` | theta/alpha power, FWER tests and connected reduction |
| `03_subject_identification.ipynb` | subject-identity control |
| `04_logistic_baselines.ipynb` | theta and alpha LR baselines |
| `05_riemann.ipynb` | Riemann TS-LDA |
| `06_svc.ipynb` | agnostic and cluster SVC |
| `07_eegnet.ipynb` | agnostic and cluster EEGNet |
| `08_st_gnn.ipynb` | graph neural network |
| `09_st_eegformer.ipynb` | pretrained transformer |
| `10_labram.ipynb` | pretrained LaBraM |
| `11_explainability.ipynb` | SHAP visualization |
| `12_results_report.ipynb` | common HTML results report |

Each notebook has one import cell at the beginning and contains no dependency
installation, path discovery or training-loop implementation. Model notebooks
load the selected task and its saved artifacts into a scientific report:
cohort and trial counts, preprocessing and split configuration, reproducible
commands, subject-level and aggregate metrics, confusion and prediction
diagnostics, model-specific visualisations, interpretation, limitations and
baseline comparisons. Changing `TASK_ID` runs the same report for another
configured task.

There is intentionally one notebook per analysis or model family, not one copy
per task. Set `TASK_ID` near the top of a model notebook and run it to produce
that task's artifacts. To generate the common results site, use:

```text
br-analysis report
```

This writes `artifacts/report/index.html`. It contains a task selector,
coverage matrix, subject-level summaries, confidence intervals,
subject-normalized confusion matrices and subject-weighted ROC/precision-recall
curves. It compares models within each task and tasks within each model.
Artifacts produced by the former development-CV protocol are labelled
`legacy` until rerun.

## Artifacts

Runs are stored under:

```text
artifacts/{task_id}/{model_id}/
  metadata.json
  metrics.csv
  predictions.npz
  shap_values.npz
  shap_values.json
```

Every run records task classes, preprocessing, exclusions, class-aligned OOF
probabilities and model metadata.

Preprocessed epochs are cached under `artifacts/_cache/datasets/`. Cache keys
include the FIF inventory, analysis window, preprocessing configuration and
implementation version. They deliberately do not include the decoding task or
channel subset, so overlapping tasks and cluster-based models reuse the same
epochs. Each subject is written separately, so interrupted preparation can be
resumed without repeating completed subjects. Completed model runs also store
a signature covering the data, model, split configuration, checkpoint and
code. A matching re-execution reuses the run; an incompatible one is
recalculated.

Cache filenames combine readable provenance with a short fingerprint:

```text
P002__band_limited__0p0-1p5__0d47a018ae12.npz
```

The prefix identifies subject, processing and time window. The fingerprint
changes when the FIF file or preprocessing parameters change, preventing an
outdated cache from being reused under a plausible-looking name.

Only completed runs appear in the report. Results formerly embedded in old
notebook outputs are not migrated automatically; each desired task/model pair
must be run once to enter the common artifact structure.

## Development checks

```text
python -m pytest
python -m ruff check src/br_analysis src/notebooks tests
python -m pip wheel . --no-deps --no-build-isolation
```

Tests cover task mappings, local paths, multiclass metrics, chronological
splits, classical and PyTorch runners, subject identification, SHAP result
contracts and notebook structure.
