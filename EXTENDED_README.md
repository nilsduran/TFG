# Extended Project README

## Purpose of this document

This document is an expanded technical README for the TFG project **Binocular Rivalry: EEG decoding of perceptual conflict with classical machine learning, Riemannian methods, deep learning, and foundation models**.

It is meant to be a thesis-writing basis, not the thesis itself. It summarizes the scientific context, the computational objectives, the project structure, the notebooks, the experiments, the methodological concepts, and the main results currently grounded in the repository code and notebook outputs. The writing follows the structure seen in the example bachelor theses provided by the user: context and problem, background, research design, methodology, experiments, results, limitations, and conclusions.

The real analysis data are assumed to live in the Google Drive project layout used throughout the notebooks:

```text
TFG_Nils_2026/
  02_Data/01_Preprocessed_epochs/
  04_Anàlisi/
```

The repository contains the code, notebooks, documentation, local helper package, generated text notes, and some cached or example resources. The small `data/` folder in this workspace should not be interpreted as the full dataset used in the work.

## Grounding and sources used

The descriptions below are based on:

- The existing [README.md](README.md), which already summarizes the project and notebook pipeline.
- The notebooks in [src/notebooks/](src/notebooks/), including their stored markdown, code, and cached output summaries.
- The local helper library in [src/br_analysis/](src/br_analysis/).
- The paper-derived notes in [docs/markdown/00_BR.md](docs/markdown/00_BR.md), [docs/markdown/01_Electroencephalography.md](docs/markdown/01_Electroencephalography.md), [docs/markdown/02_Review_ML4EEG.md](docs/markdown/02_Review_ML4EEG.md), and [docs/markdown/03_Classification_methods.md](docs/markdown/03_Classification_methods.md).
- The current thesis draft in [docs/main.tex](docs/main.tex).
- The example theses provided in `C:\Users\nils\Downloads`, used only to infer a reasonable document structure.

No claim in this document should be read as a new experimental result unless it appears in the repository notebooks or code. Where a result is exploratory or based on a cached run, it is described as such.

## Project context

Binocular rivalry (BR) is a perceptual phenomenon that appears when incompatible visual images are presented simultaneously to the two eyes. Instead of fusing both inputs into one stable percept, the visual system alternates between possible interpretations: one image dominates, the other dominates, or a mixed/piecemeal percept is experienced. This makes BR useful for studying how the brain resolves perceptual conflict and how conscious perception is selected when the sensory input itself remains ambiguous.

The project builds on the article and experimental paradigm described in the repository notes, especially the study "Binocular Rivalry: Evaluating the Role of Theta Power as a Neural Index of Conflict". The central scientific question is whether the neural signatures usually associated with cognitive conflict also appear during binocular rivalry when motor-report confounds are reduced. The underlying hypothesis is that rivalry between incompatible visual representations recruits conflict monitoring and attentional control mechanisms.

The original paper emphasizes two frequency bands and two spatial regions:

- **Fronto-medial theta power**, approximately 5-7 Hz, associated with conflict monitoring and often linked to medial frontal regions such as the anterior cingulate cortex.
- **Parieto-occipital alpha power**, approximately 8-12 Hz, associated with visual attention, inhibition, and sensory gating. In this context, lower alpha power can be interpreted as stronger visual engagement or reduced inhibition.

This TFG begins from the original paper's statistical analysis and then expands it. The first aim is to reproduce the paper's main theta/alpha findings as a grounding step. The second aim is methodological: instead of only using the paper's one-dimensional temporal clustering within predefined regions, the project extends the analysis to two-dimensional spatiotemporal clustering across channels and time. The third aim is to ask whether single-trial EEG decoding and modern machine learning can distinguish perceptual/stimulus conditions related to conflict. The project therefore sits between cognitive neuroscience and applied machine learning: the models are not only evaluated for accuracy, but also for whether their spatial, temporal, and spectral behavior is compatible with the known BR conflict framework.

## Main objectives

The project has four main objectives.

First, it aims to **replicate the main results of the original binocular-rivalry paper**. This means checking whether the expected theta and alpha effects can be recovered in the available EEG data and whether the project reproduces the basic interpretation of fronto-medial theta and parieto-occipital alpha as conflict-related sensor-level signatures.

Second, it expands the paper's statistical approach through **two-dimensional spatiotemporal clustering**. Where the original analysis focused on temporal clustering within predefined regions or signals, this project searches for structured channel-time regions in theta and alpha bands for both C vs IP and IM vs IP. This makes clustering one of the core methodological contributions, not just a preprocessing step.

Third, it extends the analysis from condition-level statistics to **single-trial decoding**. This includes interpretable classical baselines such as logistic regression, Riemannian covariance classification with LDA, and SVC clustering, as well as deep learning and pretrained EEG models such as EEGNet, a spatio-temporal graph neural network, ST-EEGFormer, and LaBraM.

Fourth, it interprets the full pattern of results. The thesis-relevant aim is not just "which model wins", but whether the replicated statistics, the spatiotemporal clusters, the decoding benchmarks, and the explainability outputs tell a coherent story about the detectability of binocular-rivalry conflict in EEG.

## Experimental paradigm

The task uses an onset rivalry design. In continuous rivalry paradigms, subjects continuously report changes in perception, which can mix perceptual activity with motor preparation, button pressing, and introspection. In the onset design, the stimulus period is separated from the later response screen. This makes the EEG during stimulus presentation easier to interpret as perceptual processing rather than direct motor reporting.

The trial structure summarized by the project is:

```text
blank / fixation -> stimulus fade-in -> stimulus presentation -> blank jitter -> response screen
```

The important conditions in the code are:

- **Congruent (C)**: compatible or identical visual stimulation, with no interocular conflict.
- **Incongruent Pure (IP)**: incompatible images are presented, but the subject reports a pure percept. This is interpreted as a case where conflict was resolved in favor of one percept.
- **Incongruent Mixed (IM)**: incompatible images are presented and the subject reports a mixed percept. This is interpreted as unresolved, unstable, or piecemeal perceptual competition.

The code defines two binary decoding tasks in [src/br_analysis/config.py](src/br_analysis/config.py):

```text
c_vs_ip:
  class 0 = Congruent, event codes [1, 2]
  class 1 = Incongruent_Pure, event codes [3, 4, 5, 6]
  max subjects = 24

im_vs_ip:
  class 0 = Incongruent_Mixed, event codes [7, 8]
  class 1 = Incongruent_Pure, event codes [3, 4, 5, 6]
  max subjects = 29
```

The **C vs IP** task is one of the two central contrasts. Both conditions are reported as pure percepts, but only IP contains physical interocular conflict. A successful classifier should therefore be sensitive to neural differences related to conflict resolution rather than only to the reported percept category.

The **IM vs IP** task is also central, but it targets a different question. Both classes are incongruent trials, so the physical stimulus class is closer, but the reported percept differs. This contrast asks whether EEG contains enough information to distinguish unresolved/mixed rivalry from resolved/pure rivalry.

## EEG concepts needed for the thesis

EEG measures scalp voltage fluctuations generated mainly by synchronized postsynaptic activity in large neuronal populations. It has high temporal resolution, which is ideal for studying fast perceptual dynamics, but it has limited spatial resolution because signals are mixed by volume conduction through the brain, skull, and scalp.

An EEG trial can be represented as:

```math
X_i \in \mathbb{R}^{C \times T}
```

where `C` is the number of electrodes and `T` is the number of time samples. In this project, many models use either all 60 EEG channels or subsets of channels selected from theta/alpha clusters. The original recordings are sampled at 500 Hz in the exploratory notebooks, while deep learning caches typically resample to 128 Hz to reduce computational cost.

The discrete EEG signal is a sampled version of an underlying continuous voltage signal:

```math
x_c[n] = x_c(n / f_s),\quad n = 0,\ldots,T-1
```

where `f_s` is the sampling frequency. The Nyquist-Shannon condition says that a signal with no energy above `f_max` can in principle be reconstructed if:

```math
f_s > 2 f_{max}
```

This matters because the project studies theta and alpha activity directly, but also uses broader 1-40 Hz inputs for some models. Downsampling to 128 Hz is still above the Nyquist limit for 40 Hz, while reducing memory and training cost.

The main signal-processing concepts are:

- **Epochs**: short trial-aligned EEG segments around stimulus onset.
- **Filtering**: restricting the signal to a frequency band, such as theta 5-7 Hz or alpha 8-12 Hz.
- **Time-frequency analysis**: estimating how power changes over time and frequency.
- **Envelope / Hilbert amplitude**: after band-pass filtering, the analytic signal gives an amplitude envelope that approximates instantaneous oscillatory strength.
- **Power**: often proportional to squared amplitude. In many EEG analyses, power is baseline-corrected and sometimes expressed in decibels.
- **ERP**: an event-related potential obtained by averaging voltage time series across trials. ERPs capture activity phase-locked to the event, whereas induced oscillations may not be phase-locked.
- **Topography**: the spatial distribution of EEG activity over scalp electrodes. Topographies are useful but should not be overinterpreted as direct source locations.

For an event-related potential, if condition `k` has `N_k` trials, the average waveform at channel `c` is:

```math
ERP_k(c,t) = \frac{1}{N_k}\sum_{i:y_i=k} X_i(c,t)
```

This averaging improves phase-locked signal-to-noise ratio, but it discards single-trial variability. The decoding notebooks do the opposite: they preserve single trials and ask whether a classifier can infer the condition from each individual epoch.

Filtering can be written abstractly as a convolution:

```math
x_{band}(t) = (h_{band} * x)(t)
```

where `h_band` is a band-pass filter. In frequency space, this corresponds to:

```math
X_{band}(f) = H_{band}(f)X(f)
```

with `H_band(f)` retaining the desired frequencies and attenuating others. The notebooks use theta, alpha, theta+alpha, broadband, and task-specific cluster windows depending on the experiment.

A simple power feature for one trial, one channel set, and one time window can be written as:

```math
P_i = \frac{1}{|C||T|}\sum_{c \in C}\sum_{t \in T} x_i(c,t)^2
```

For Hilbert-envelope features, if `z_i(c,t)` is the analytic signal after band-pass filtering:

```math
A_i = \frac{1}{|C||T|}\sum_{c \in C}\sum_{t \in T} |z_i(c,t)|
```

The analytic signal is:

```math
z(t)=x_{band}(t)+j\mathcal{H}\{x_{band}(t)\}
```

where `\mathcal{H}` is the Hilbert transform. The instantaneous amplitude is `|z(t)|`, and instantaneous phase is:

```math
\phi(t)=\arg z(t)
```

Some exploratory analyses use time-frequency representations. A short-time Fourier transform can be written as:

```math
STFT_x(\tau,f)=\sum_t x(t)w(t-\tau)e^{-j2\pi ft}
```

and the corresponding power is:

```math
P(\tau,f)=|STFT_x(\tau,f)|^2
```

For wavelet-like reasoning, a Morlet transform has a similar objective: localize oscillatory power in both time and frequency. The unavoidable trade-off is that narrow frequency resolution requires longer windows, while precise temporal resolution requires shorter windows. This is why the thesis should avoid interpreting a cluster onset as an exact neural onset.

Many EEG power plots use baseline correction. If `P(t,f)` is trial or condition power and `\bar{P}_{base}(f)` is mean baseline power, a common decibel transform is:

```math
P_{dB}(t,f)=10\log_{10}\left(\frac{P(t,f)}{\bar{P}_{base}(f)}\right)
```

Finally, many deep models standardize each trial/channel over time:

```math
\tilde{x}_{i,c,t}=\frac{x_{i,c,t}-\mu_{i,c}}{\sigma_{i,c}+\epsilon}
```

This makes the model less sensitive to absolute amplitude scale and more sensitive to temporal shape. It can help generalization, but it may also remove physiologically meaningful amplitude information if used indiscriminately. This is why normalization is tested explicitly in Experiment 1 and Experiment 8.

These features are simple, but they are directly connected to the neuroscientific hypotheses about theta and alpha activity.

## Binocular rivalry concepts needed for the thesis

BR is useful because the stimulus and the percept can be partially dissociated. In ordinary vision, a stimulus usually maps to a stable percept. In BR, incompatible monocular inputs force competition between alternatives, so the same physical input can lead to different perceptual reports.

This project relies on three interpretations:

- **Interocular conflict**: IP and IM trials contain incompatible left-eye and right-eye signals. C trials do not.
- **Conflict resolution**: IP is treated as an incongruent trial where the system resolved competition into one dominant percept.
- **Unstable or mixed perception**: IM is treated as a case where competition was not resolved into one pure percept during the reporting-relevant interval.

The theoretical expectation is that IP should differ from C in fronto-medial theta and parieto-occipital alpha because IP requires resolving interocular conflict. The IM vs IP contrast tests a complementary question: whether mixed and pure perceptual outcomes differ when both trial types contain interocular conflict. The paper notes that mixed percepts might reflect stronger or less resolved conflict, but the project results show that decoding IM vs IP is generally harder and often near chance.

## Statistical and machine learning concepts

This section should support the thesis methods without becoming a second textbook. The essential point is that the project treats decoding as a controlled comparison of EEG representations, not as a generic prediction challenge.

### Binary decoding and class balance

Most analyses use binary classification on single EEG trials:

```math
f_\theta(X_i) \rightarrow \hat{y}_i \in \{0, 1\}
```

The two central contrasts are **C vs IP** and **IM vs IP**. They should both be presented as scientifically important: C vs IP asks whether physical interocular conflict can be detected when the reported percept is pure, while IM vs IP asks whether resolved and mixed incongruent percepts differ in EEG.

Because the natural class distribution is uneven, trials are balanced within subject by undersampling the larger class:

```math
n_{keep} = \min(n_0,n_1)
```

This makes chance performance close to 0.5 and prevents class prevalence from dominating the models, but it also discards trials and increases variance. Subject selection therefore uses two practical criteria: enough trials in the minority class, usually IP, and sufficient condition separation according to Cohen's d. The thesis should explain this selection logic, but it does not need a long standalone treatment.

### Validation and chance testing

Cross-validation is mainly relevant in the decoding chapter. The key thesis point is that EEG trials can share slow drifts, block effects, fatigue, impedance changes, or subject-specific patterns, so validation cannot be treated casually. The project uses temporal/grouped splitting where possible and keeps preprocessing inside folds to reduce leakage.

Some notebooks also use label permutation to estimate empirical chance distributions:

```math
p = \frac{1 + \sum_{p=1}^{P} I(A_p \ge A_{real})}{P+1}
```

This is useful because modest above-chance accuracies are only interpretable if compared with a reasonable null distribution. Standard metrics such as accuracy, balanced accuracy, confusion matrices, ROC, and PR curves are reporting tools rather than central thesis concepts.

### Statistical correction

False discovery rate correction appears in some comparison or visualization code, especially for multiple pairwise/model comparisons. However, unless a final thesis result depends on an FDR-significant finding, FDR should remain a short methodological note. The main statistical correction framework worth foregrounding is the cluster-based permutation analysis, because it directly supports the clustering chapter.

### Cluster-based permutation testing

Cluster-based permutation testing is important enough to motivate its own thesis section or chapter. Instead of interpreting isolated channel-time samples, the method groups neighboring samples into spatiotemporal clusters and compares their cluster mass against a permutation distribution.

The basic logic is:

1. Compute a statistic for each channel-time cell.
2. Apply an initial cluster-forming threshold.
3. Connect neighboring cells in time and scalp space.
4. Compute a cluster mass.
5. Permute labels many times and keep the maximum cluster mass per permutation.
6. Assign corrected p-values to observed clusters.

This is appropriate for EEG because plausible effects should usually extend over neighboring electrodes and time samples. A cluster boundary should still not be interpreted as an exact neural onset, because time-frequency resolution and preprocessing choices limit temporal precision.

### Classical decoding models

Logistic regression is used as a transparent baseline on compact theta/alpha features. It is useful precisely because it is simple: if a single-channel or small-ROI feature carries information, a regularized linear classifier should detect some of it.

Riemannian covariance classification is the strongest classical family in the current project. Each trial is represented by a covariance matrix, stabilized with OAS shrinkage, projected into tangent space, and classified with shrinkage LDA. This is well suited to EEG because it preserves distributed spatial relationships between sensors and frequency-filtered networks.

The SVC clustering notebook adds a later classical model that uses compact cluster-restricted features from the exported theta/alpha windows. It should be described as an additional cluster-informed classifier rather than as a completely separate methodological pillar.

### Deep learning and pretrained models

The deep-learning models test whether more flexible architectures can learn useful representations from raw or lightly processed EEG:

- **EEGNet**: compact EEG-specific convolutional model, tested in agnostic and cluster-restricted variants.
- **ST-GNN**: graph neural network using electrode geometry and temporal features.
- **ST-EEGFormer**: transformer-style EEG model adapted with LoRA.
- **LaBraM**: pretrained/foundation EEG model adapted with LoRA-style fine-tuning.

These models require careful interpretation because the dataset is small, noisy, and subject-specific. Their role in the thesis is not simply to "beat" classical methods, but to test whether model complexity and pretraining help under realistic EEG constraints.

### Explainability and attribution

Saliency and SHAP-style analyses are used as model-behavior diagnostics. They ask whether trained models rely on plausible channels, time windows, and frequency-related structure. They should not be presented as direct source localization or causal neuroscience evidence.
## Repository structure

```text
.
  README.md
  EXTENDED_README.md
  requirements.txt
  kaggle_requirements.txt
  docs/
    main.tex
    markdown/
    bibliografia/
    media/
  src/
    br_analysis/
    notebooks/
    scripts/
```

The most important code package is [src/br_analysis/](src/br_analysis/). It turns repeated notebook logic into reusable components:

- [config.py](src/br_analysis/config.py): path detection, binary task configuration, event-code definitions, band ranges, target channel clusters, and epoch grids.
- [datasets.py](src/br_analysis/datasets.py): MNE `.fif` epoch loading, event filtering, cropping, class balancing, band/channel dataset construction, and Google Drive file-copy safeguards.
- [features.py](src/br_analysis/features.py): Hilbert mean envelope and mean power feature extractors.
- [models.py](src/br_analysis/models.py): logistic regression and Riemannian LDA model factories.
- [experiments.py](src/br_analysis/experiments.py): classical ML experiment loops, temporal/grouped CV, permutation nulls, epoch-grid analysis, time-window maps, and paired map statistics.
- [deep.py](src/br_analysis/deep.py): raw EEG dataset loading for neural models, resampling, filtering, and adjacency construction.
- [torch_models.py](src/br_analysis/torch_models.py): EEGNet, ST-GNN, LoRA layers, ST-EEGFormer-style model components, and LaBraM-style adaptation components.
- [pretrained.py](src/br_analysis/pretrained.py): pretrained model loading and LoRA injection helpers.
- [torch_training.py](src/br_analysis/torch_training.py): PyTorch training loops, standardization, temporal validation, refitting, cross-validation metrics, ROC/PR plotting, and learning/time maps.
- [plotting.py](src/br_analysis/plotting.py): reusable visualization helpers.

The script [src/scripts/br.py](src/scripts/br.py) contains Manim animations for explaining the BR trial timeline, stereoscope setup, congruent/incongruent stimuli, and response alternatives.

## Notebook pipeline

The notebooks form a workflow from exploration to model comparison. The thesis should describe this pipeline by analysis role, not by giving every notebook equal space.

### Dataset exploration and subject selection

[01_exploració_dades.ipynb](src/notebooks/01_exploració_dades.ipynb) loads the EEG epochs and checks the behavioral and signal structure of the dataset. It documents the natural imbalance between C, IP, and IM trials; subject-level perceptual biases; eye/color dominance; usable trial counts; ERP/time-frequency summaries; and possible fatigue or session-progress effects.

The most thesis-relevant output is the subject-selection logic. Subjects are not selected only by availability: selection considers the number of minority-class trials and an effect-size criterion based on Cohen's d. This should be explained generally, with detailed tables moved to an appendix if needed.

### Representation search and exploratory experiments

[02_experiments.ipynb](src/notebooks/02_experiments.ipynb) is a broad representational testing framework rather than one final model. It covers preprocessing/normalization ablations, spatial-temporal-spectral representation diagnostics, classical ML baselines, CSP/FBCSP/Riemannian comparisons, PLV/coherence connectivity features, EEGNet and deep-learning ablations, transfer learning, joint configuration search, controlled contribution ablations, generalization checks, and subject-identity controls.

For the thesis, this notebook should be used to explain how the project narrowed down the analysis space. Important takeaways include:

- preprocessing and representation choices matter, but they should not dominate the thesis narrative;
- Riemannian/covariance representations repeatedly appear strong;
- PLV/connectivity features are part of the work but are not a major final result if they remain close to chance;
- subject identity is easy for EEG models to learn and should be discussed as a control and ethics issue.

### Simple targeted decoding

[03_chance_level_c_ip.ipynb](src/notebooks/03_chance_level_c_ip.ipynb) and [07_chance_level_im_ip.ipynb](src/notebooks/07_chance_level_im_ip.ipynb) test simple theta/alpha logistic-regression baselines. They use targeted features such as FCz-theta and Oz-alpha, epoch grids, time-window maps, and permutation chance estimates.

These notebooks are useful because they establish transparent baselines. They show that isolated features carry only modest information, with alpha generally more useful than single-channel theta in the stored outputs. The detailed grid counts can be summarized in the results chapter and placed in an appendix if necessary.

### Riemannian covariance decoding

[04_riemann_c_ip.ipynb](src/notebooks/04_riemann_c_ip.ipynb) and [08_riemann_im_ip.ipynb](src/notebooks/08_riemann_im_ip.ipynb) implement covariance-based decoding with Riemannian tangent-space projection and shrinkage LDA. This is one of the most important model families because it preserves multichannel covariance structure instead of reducing trials to one or two scalar power features.

The thesis should connect this method to the EEG interpretation: discriminative information may be distributed across channels and frequency networks rather than located in a single electrode.

### Experiment-guided configurations

[05_experiment_guided_config_c_ip.ipynb](src/notebooks/05_experiment_guided_config_c_ip.ipynb) and [09_experiment_guided_config_im_ip.ipynb](src/notebooks/09_experiment_guided_config_im_ip.ipynb) apply configurations selected by the broader experiment-search notebook and evaluate them with development CV and a reserved holdout.

These notebooks are useful for explaining how the project moved from search to more constrained evaluation. The thesis should be careful to describe configuration search as exploratory/model-selection evidence, not as independent confirmation by itself.

### Visualization and interpretive figures

[06_visualitzacions.ipynb](src/notebooks/06_visualitzacions.ipynb) collects figure-oriented analyses: searchlight maps, subject-selection views, topographies, temporal maps, alpha-vs-theta comparisons, ERP images, Riemannian projections, connectivity graphs, and other exploratory visualizations.

This notebook is mainly a source of thesis figures and interpretive support. It should not be described as a separate experimental result unless a specific figure is used to support a claim.

### Clustering

[10_clustering.ipynb](src/notebooks/10_clustering.ipynb) deserves substantial thesis space. It performs data-driven spatiotemporal cluster analysis for both C vs IP and IM vs IP in theta and alpha bands, compares paper-inspired ROIs with data-driven clusters, and exports cluster definitions for later focused models.

The exported clusters in the stored output are:

```json
{
  "c_vs_ip": {
    "theta": {"channels": ["Fz", "FCz"], "tmin": 1.171875, "tmax": 1.2421875},
    "alpha": {"channels": ["FC3", "FC1", "FC2", "C3", "C1", "Cz", "C2", "CP3", "CP1", "CPz", "CP2", "CP4", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"], "tmin": 0.375, "tmax": 1.2421875}
  },
  "im_vs_ip": {
    "theta": {"channels": ["Fz", "FCz", "Cz"], "tmin": 1.1953125, "tmax": 1.2421875},
    "alpha": {"channels": ["TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3"], "tmin": 0.2578125, "tmax": 0.421875}
  }
}
```

Stored outputs suggest that C vs IP alpha effects are among the more stable sensor-level findings, while IM vs IP clusters are less stable and more sensitive to ROI/time-frequency choices. This should be framed carefully: clustering is central for interpretation and feature restriction, but cluster-selected regions can introduce circularity if reused for final decoding without separation between selection and evaluation.

### Deep learning and pretrained models

[11_EEGNet_c_ip.ipynb](src/notebooks/11_EEGNet_c_ip.ipynb), [17_EEGNet_im_ip.ipynb](src/notebooks/17_EEGNet_im_ip.ipynb), [12_ST-GNN_c_ip.ipynb](src/notebooks/12_ST-GNN_c_ip.ipynb), [18_ST-GNN_im_ip.ipynb](src/notebooks/18_ST-GNN_im_ip.ipynb), [13_ST-EEGFormer_c_ip.ipynb](src/notebooks/13_ST-EEGFormer_c_ip.ipynb), [19_ST-EEGFormer_im_ip.ipynb](src/notebooks/19_ST-EEGFormer_im_ip.ipynb), [14_LaBraM_c_ip.ipynb](src/notebooks/14_LaBraM_c_ip.ipynb), and [20_LaBraM_im_ip.ipynb](src/notebooks/20_LaBraM_im_ip.ipynb) evaluate neural and pretrained EEG models.

The most important thesis-level point is that deep/foundation models do not automatically outperform classical methods in this dataset. EEGNet agnostic is a useful neural baseline; cluster-restricted EEGNet does not clearly improve cohort performance in the stored outputs; ST-GNN, ST-EEGFormer, and LaBraM are useful exploratory comparisons but require cautious reporting, especially where cached IM vs IP predictions are incomplete.

### Cross-model comparison

[15_cross_model_comparison_c_ip.ipynb](src/notebooks/15_cross_model_comparison_c_ip.ipynb) and [21_cross_model_comparison_im_ip.ipynb](src/notebooks/21_cross_model_comparison_im_ip.ipynb) collect the main benchmark comparisons.

For C vs IP, the stored comparison ranks Riemannian covariance highest, followed by LR-alpha and EEGNet agnostic, with larger neural/foundation models not clearly improving performance. For IM vs IP, the stored comparison is more limited because some expected deep-model prediction caches were missing, but Riemannian covariance again appears strongest among the available models.

Pairwise tests and FDR-corrected heatmaps can be mentioned if used in final figures, but they do not need a long conceptual section unless they drive a thesis conclusion.

### SVC clustering

[23_SVC_clustering_c_ip_im_ip.ipynb](src/notebooks/23_SVC_clustering_c_ip_im_ip.ipynb) is not reflected in older versions of this README. It trains an SVC model on compact features tied to the exported cluster windows for both C vs IP and IM vs IP, writes prediction caches, and integrates with the later explainability/comparison utilities.

This notebook belongs with the clustering and classical decoding story: it asks whether cluster-restricted theta/alpha summaries can support a classifier more flexibly than logistic regression while staying more interpretable than the deep models.

### Explainability

[16_explainability_c_ip.ipynb](src/notebooks/16_explainability_c_ip.ipynb), [22_explainability_im_ip.ipynb](src/notebooks/22_explainability_im_ip.ipynb), and [src/br_analysis/explainability.py](src/br_analysis/explainability.py) centralize attribution and model-behavior summaries across deep and cluster-informed models.

Their thesis role is supporting rather than primary: they help ask whether models rely on plausible channels and time windows, but sparse or incomplete stored outputs should be described as interpretability infrastructure unless rerun and finalized.
## Main results and interpretation

### Behavioral and dataset structure

The dataset is not balanced naturally. IM trials are much more frequent than IP trials, and subject-level perceptual biases are common. The project handles this through subject selection and random undersampling in the decoding tasks. This is not just a preprocessing detail: it is central to the validity of the thesis, because models could otherwise learn class prevalence or temporal drift rather than neural signatures of conflict.

The selected cohorts used by the main decoding notebooks are:

- 24 subjects for C vs IP.
- 29 subjects for IM vs IP.

### C vs IP is empirically more robust

Across notebooks, C vs IP tends to produce stronger decoding than IM vs IP. This fits the scientific design: C vs IP compares no-conflict pure perception with conflict-resolved pure perception. The perceptual report is similar, but the physical and neural conflict context differs.

The best model in the C vs IP cross-model comparison is Riemannian covariance + LDA with mean accuracy around 0.566. This is not high in conventional computer vision terms, but for EEG single-trial decoding it is meaningful if validated carefully. The result indicates that conflict-related information is weak, noisy, and variable across subjects, but not absent.

### Riemannian covariance features outperform most alternatives

The Riemannian model is best for C vs IP and also best among available IM vs IP benchmarks. This suggests that the discriminative information is not captured only by average power at one electrode. Instead, it may live in covariance patterns across theta and alpha networks.

This interpretation is compatible with EEG physiology: scalp EEG is spatially mixed, and cognitive/perceptual states often appear as distributed changes in relationships between sensors rather than isolated point activations.

### Configuration search reinforces the importance of preprocessing

The current Experiment 8 results show that preprocessing and representation choices matter as much as model choice. C vs IP favors a broadband Riemannian configuration with trial-wise z-scoring and all channels. IM vs IP favors a more constrained alpha/ROI classical-ML setup with robust scaling and Welch PSD features.

This is useful for the thesis because it prevents a simplistic "deep learning vs classical ML" narrative. The more accurate framing is that EEG decoding is a pipeline problem: filtering, window selection, spatial selection, normalization, feature extraction, model family, and validation all interact. A model can only learn what the representation preserves.

### Alpha features appear more useful than single-channel theta in simple LR

In the simple logistic regression notebooks, Oz-alpha generally performs better than FCz-theta, especially for C vs IP. This does not disprove the fronto-medial theta hypothesis from the paper. It means that, under this specific decoding setup, single-channel theta envelope alone is weak as a classifier feature, while alpha carries more class-discriminative information.

The Riemannian results suggest that theta may still contribute as part of a multichannel covariance representation.

### Deep learning does not automatically win

EEGNet agnostic performs reasonably for C vs IP, but it does not beat Riemannian covariance. EEGNet clustering performs worse than agnostic EEGNet for C vs IP in the stored output. ST-GNN and ST-EEGFormer are close to chance or modestly above it. LaBraM does not improve performance in the cached C vs IP results.

This is an important thesis conclusion: model complexity is not equivalent to better EEG decoding. The dataset is small, noisy, and subject-specific. Classical methods with strong inductive biases can outperform larger neural models when the signal is low-SNR and the training set is limited.

### IM vs IP remains difficult

IM vs IP is harder than C vs IP. The physical stimulus class is incongruent in both cases, and the difference lies in the reported perceptual outcome. The available results show weak decoding, with Riemannian accuracy around 0.532 and logistic theta/alpha near chance.

This can be interpreted in several non-exclusive ways:

- The neural difference between mixed and pure incongruent trials is weaker than the difference between congruent and incongruent pure trials.
- The relevant signals may not be captured by the selected time windows, channels, or bands.
- Subject reports of mixed perception may include heterogeneous experiences.
- Trial counts and class imbalance make the contrast statistically harder.

### Interpretability is necessary, not optional

Because performance is modest, the thesis should not argue only from accuracy. It should combine:

- Accuracy and permutation/null comparisons.
- Spatial maps and cluster definitions.
- Temporal windows.
- Band-specific interpretation.
- Cross-model agreement.
- Subject-level variability.

The strongest thesis argument is therefore not "the model classifies BR perfectly", but rather: "carefully validated decoding reveals weak but structured EEG information related to binocular-rivalry conflict, and covariance-based representations capture this structure better than isolated band-power or large neural models in the current setting."

### Subject identity as an ethics/control issue

Experiment 11's subject-identification control reaches extremely high accuracy in the cached output. This is not a target result for the main decoding story, but it is important for validation and ethics: EEG contains stable person-specific structure, so cohort-level models must be interpreted carefully and neural data should be treated as sensitive biometric data. This point fits naturally in an ethics section, with a short methodological reminder in the decoding discussion.

## Limitations and risks

The project has several limitations that should be explicitly discussed in the thesis.

First, the decoding accuracies are modest. This is expected for noisy single-trial EEG, but it means conclusions must be cautious.

Second, subject variability is large. Some subjects, such as P057 in several notebooks, show much stronger decoding than the cohort average. The thesis should avoid generalizing from a single subject.

Third, several deep-learning IM vs IP notebooks still lack exported prediction caches in the cross-model comparison. Before final thesis submission, those models should either be rerun and exported consistently or clearly excluded from the final IM vs IP benchmark.

Fourth, clustering can introduce circularity if the same data are used both to choose clusters and to estimate final model performance. The current project uses clustering mainly for interpretability and focused model variants, but the thesis should describe this risk and distinguish exploratory cluster selection from confirmatory testing.

Fifth, EEG topographies do not directly identify brain sources. Fronto-medial and parieto-occipital language should be interpreted as sensor-level regions unless source localization is performed.

Sixth, pretrained EEG foundation models require careful adaptation. Their underperformance here may reflect limited data, checkpoint mismatch, channel-layout differences, fine-tuning choices, or task mismatch rather than a general failure of foundation models.

Seventh, the subject-identification control in Experiment 11 shows that EEG contains a strong subject fingerprint. This is expected, but it means high within-cohort performance can be misleading if the validation split allows subject identity to help the model. Any final thesis claim about condition decoding should distinguish within-subject, within-cohort, LOSO, and cross-condition settings.

## Thesis structure suggested by this README

A natural thesis structure would be:

1. **Introduction**: binocular rivalry, perceptual conflict, EEG decoding, motivation for comparing decoding approaches.
2. **Theoretical background**: binocular rivalry, EEG, theta/alpha oscillations, conflict monitoring, and machine learning for EEG.
3. **Dataset, experimental design, and exploratory analysis**: onset rivalry protocol, event codes, task construction, subject selection, class imbalance, behavioral biases, and concise preprocessing.
4. **Clustering and statistical characterization**: paper-inspired ROIs, data-driven spatiotemporal clusters, theta/alpha windows, and circularity risks.
5. **Decoding methods**: logistic regression, Riemannian covariance, SVC clustering, EEGNet, ST-GNN, ST-EEGFormer, LaBraM, cross-validation, and chance testing.
6. **Results**: EDA summary, clustering results, C vs IP decoding, IM vs IP decoding, cross-model comparison, and interpretability evidence.
7. **Discussion**: what both contrasts imply, why covariance methods worked well, why deep models did not dominate, and what the limitations mean.
8. **Ethics**: EEG privacy, subject identity, anonymization, responsible claims about neural decoding, and reproducibility.
9. **Conclusions and future work**: final answer to the objectives, improvements for reruns, stronger validation, richer features, and subject-transfer analysis.

## Conclusions

This project investigates whether EEG signals recorded during onset binocular rivalry contain decodable information about perceptual conflict and conflict resolution. The two central contrasts are C vs IP, where pure percept reports differ in physical interocular conflict, and IM vs IP, where incongruent trials differ in whether the reported percept is mixed or pure.

The current evidence supports a cautious positive answer. The signal is weak, but not random: Riemannian covariance classification reaches the best stored performance, and alpha/theta cluster analyses identify biologically plausible time-channel regions. C vs IP is empirically more robust in the current outputs, while IM vs IP remains harder to decode and should be interpreted with more caution.

The methodological conclusion is also clear. For this dataset, carefully designed classical EEG methods outperform or match more complex deep learning and pretrained models. Riemannian covariance features are especially valuable because they preserve distributed spatial structure while remaining appropriate for small EEG datasets. Deep models remain useful as exploratory tools and for interpretability comparisons, but they require more data, careful tuning, and stricter cache/reproducibility checks before being treated as final evidence.

The thesis should therefore frame the project as a rigorous EEG decoding study rather than a high-accuracy prediction challenge. Its contribution is the systematic comparison of interpretable and modern ML approaches for binocular-rivalry EEG, showing that conflict-related perceptual information is present but subtle, distributed, and highly sensitive to validation design.

