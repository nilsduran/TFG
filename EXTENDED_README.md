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

This TFG does not simply reproduce the paper's statistical analysis. Instead, it asks whether single-trial EEG decoding and modern machine learning can distinguish perceptual/stimulus conditions related to conflict. The project therefore sits between cognitive neuroscience and applied machine learning: the models are not only evaluated for accuracy, but also for whether their spatial, temporal, and spectral behavior is compatible with the known BR conflict framework.

## Main objectives

The project has four main objectives.

First, it explores the EEG dataset and the behavioral structure of the task. This includes checking class imbalance, subject-level dominance biases, color and eye biases, possible session fatigue, and the number of usable trials per subject. This step is necessary because EEG decoding can be easily distorted by class imbalance, temporal drift, or subject-specific response patterns.

Second, it builds classical machine learning baselines using interpretable features. These include logistic regression on theta/alpha band-envelope features and Riemannian covariance classification with LDA. These methods are important because they are relatively transparent and well matched to EEG: they make explicit assumptions about oscillatory power, covariance structure, and low-sample classification.

Third, it evaluates deep learning and pretrained EEG models, including EEGNet, a spatio-temporal graph neural network, ST-EEGFormer, and LaBraM. These models test whether more expressive architectures can learn better representations from raw or lightly processed EEG trials.

Fourth, it compares models and interprets their behavior. The work includes cross-model benchmarking, saliency and SHAP-style explainability notebooks, cluster-based spatiotemporal analysis, and summary plots. The thesis-relevant aim is not just "which model wins", but what the performance pattern says about the detectability of binocular-rivalry conflict in EEG.

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

The **C vs IP** task is the main hypothesis-driven contrast. Both conditions are reported as pure percepts, but only IP contains physical interocular conflict. A successful classifier should therefore be sensitive to neural differences related to conflict resolution rather than only to the reported percept category.

The **IM vs IP** task is more exploratory. Both classes are incongruent trials, so the physical stimulus class is closer, but the reported percept differs. This contrast asks whether EEG contains enough information to distinguish unresolved/mixed rivalry from resolved/pure rivalry.

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

The main theoretical expectation is that IP should differ from C in fronto-medial theta and parieto-occipital alpha because IP requires resolving interocular conflict. The IM vs IP contrast is more uncertain. The paper notes an exploratory expectation that mixed percepts might reflect stronger conflict, but the project results show that decoding IM vs IP is generally harder and often near chance.

## Statistical and machine learning concepts

### Binary classification

Most notebooks formulate the task as binary classification:

```math
f_\theta(X_i) \rightarrow \hat{y}_i \in \{0, 1\}
```

where `X_i` is one EEG trial and `y_i` is the condition label. Accuracy is:

```math
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
```

Balanced accuracy is useful when fold-level labels are not perfectly balanced:

```math
BalancedAccuracy = \frac{1}{2}\left(\frac{TP}{TP+FN}+\frac{TN}{TN+FP}\right)
```

The confusion matrix is also important because a model can have apparently reasonable accuracy while favoring one class:

```math
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
```

For probabilistic models, the ROC curve summarizes the trade-off between true positive rate and false positive rate as the decision threshold changes:

```math
TPR = \frac{TP}{TP+FN},\quad FPR = \frac{FP}{FP+TN}
```

Precision-recall curves are also useful because they focus on the positive class:

```math
Precision = \frac{TP}{TP+FP},\quad Recall = \frac{TP}{TP+FN}
```

Because the project undersamples classes to balance the two labels, chance accuracy is approximately 0.5. However, the notebooks still use permutation or analytical chance estimates because small sample sizes, cross-validation design, and temporal grouping can move the practical null distribution away from a naive 50%.

Undersampling is done per subject to avoid letting a naturally larger class dominate the loss or the decision boundary. If a subject has `n_0` trials in class 0 and `n_1` trials in class 1, the balanced number of retained trials per class is:

```math
n_{keep} = \min(n_0,n_1)
```

This improves fairness between labels, but it discards data and increases variance. The epoch-grid notebooks explicitly study how decoding changes with the number of available balanced trials.

### Logistic regression

Logistic regression is used as an interpretable baseline. Given feature vector `x`, the model estimates:

```math
p(y=1|x) = \sigma(w^\top x + b)
```

where:

```math
\sigma(a) = \frac{1}{1 + e^{-a}}
```

The model is trained by minimizing cross-entropy with L2 regularization:

```math
L(w,b) =
-\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)] + \lambda ||w||_2^2
```

The project uses `SafeLogisticRegressionCV`, a wrapper around scikit-learn logistic regression CV that adapts to small class counts. This matters because some subjects have few valid trials after filtering and balancing.

The regularization strength is usually represented by `C` in scikit-learn, where larger `C` means weaker regularization:

```math
C \propto \frac{1}{\lambda}
```

For EEG, regularization is not a cosmetic detail. With few trials and many correlated features, an unregularized linear model can fit noise. L2 regularization shrinks weights smoothly, reducing variance while keeping all features available.

### Cross-validation and temporal leakage

EEG trials are not independent in the same way as shuffled images from a large dataset. Consecutive trials can share slow drifts, fatigue, impedance changes, attentional state, or block-level effects. A random train/test split can therefore overestimate performance by placing temporally adjacent trials into both training and test folds.

The project implements grouped and chronological splitting in [src/br_analysis/experiments.py](src/br_analysis/experiments.py) and [src/br_analysis/torch_training.py](src/br_analysis/torch_training.py). The intent is:

- Keep class proportions balanced across folds.
- Keep nearby trials grouped when possible.
- Reserve the latest portion of data as a stricter holdout/development exclusion.
- For cohort-level models, avoid mixing subject-specific patterns in a way that would inflate performance.

This is thesis-relevant because validation design is one of the main risks in EEG machine learning. A small difference above chance is only meaningful if leakage is controlled.

For a fold `j`, the model is trained on `D_{train}^{(j)}` and evaluated on `D_{test}^{(j)}`:

```math
A_j = \frac{1}{|D_{test}^{(j)}|}\sum_{(x_i,y_i)\in D_{test}^{(j)}} I(f_{\theta^{(j)}}(x_i)=y_i)
```

The reported cross-validation score is:

```math
\bar{A}=\frac{1}{K}\sum_{j=1}^{K} A_j
```

The project's stronger design choice is not just using `K` folds, but controlling how examples enter folds. For within-subject data, ordered trial blocks reduce leakage from neighboring trials. For cohort-level data, subject-aware groups reduce the risk of learning subject identity rather than condition structure.

### Permutation testing

Several notebooks estimate empirical chance distributions by permuting labels:

```math
y^{(p)} = \pi_p(y)
```

The classifier is retrained on permuted labels, producing a null accuracy distribution. A practical one-sided permutation p-value can be written as:

```math
p = \frac{1 + \sum_{p=1}^{P} I(A_p \ge A_{real})}{P+1}
```

The notebooks also compare real accuracy to the 95th percentile of the surrogate distribution (`chance_q95`). This is stricter than simply checking whether accuracy is above 0.5.

### Multiple comparisons and FDR

Topographic, temporal, and model-comparison analyses can involve many tests. The project uses ideas such as false discovery rate (FDR) correction and cluster-based permutation tests to reduce false positives.

FDR controls the expected proportion of false discoveries among rejected hypotheses. The Benjamini-Hochberg procedure sorts p-values and finds a threshold:

```math
p_{(k)} \le \frac{k}{m}q
```

where `m` is the number of tests and `q` is the desired FDR level.

This matters because EEG analysis often tests many channels, times, bands, and models. If 100 independent null hypotheses are tested at `p < 0.05`, roughly five false positives are expected by chance. FDR and cluster-based permutation testing are used to keep the thesis from turning noise into a story.

### Cluster-based permutation testing

Notebook 10 uses spatiotemporal cluster-based permutation testing. Instead of testing each channel-time cell independently, adjacent significant cells are grouped into clusters. A cluster statistic, such as summed t-values, is compared to a permutation-based null distribution of maximum cluster statistics. This helps control family-wise error while respecting EEG's spatial and temporal structure.

The basic logic is:

1. Compute a statistic for each channel-time cell.
2. Apply an initial cluster-forming threshold.
3. Connect neighboring cells in time and scalp space.
4. Compute a cluster mass.
5. Permute labels many times and keep the maximum cluster mass per permutation.
6. Assign corrected p-values to observed clusters.

This is appropriate for EEG because effects are expected to occupy contiguous time windows and nearby electrodes rather than isolated single samples.

A typical cell-wise statistic for a paired subject-level contrast is:

```math
t(c,t)=\frac{\bar{d}_{c,t}}{s_{d,c,t}/\sqrt{N}}
```

where `d` is the subject-level condition difference. A cluster mass can then be defined as:

```math
M(\mathcal{C})=\sum_{(c,t)\in\mathcal{C}} t(c,t)
```

The corrected cluster p-value compares the observed mass to the permutation distribution of maximum masses:

```math
p_{cluster}=\frac{1+\sum_{b=1}^{B}I(M^{max}_b \ge M_{obs})}{B+1}
```

### Riemannian covariance classification

The Riemannian model represents each trial by its covariance matrix:

```math
\Sigma_i = \frac{1}{T-1} X_i X_i^\top
```

The covariance estimator used in the notebooks is OAS shrinkage, which stabilizes covariance estimates when `T` and the number of channels are not large enough for a raw empirical covariance to be reliable:

```math
\hat{\Sigma}_{OAS}=(1-\rho)\hat{\Sigma}+\rho\frac{\mathrm{tr}(\hat{\Sigma})}{C}I
```

Covariance matrices are symmetric positive definite (SPD) when well estimated. They live on a curved manifold rather than ordinary Euclidean vector space. Distances on this manifold can be measured with the affine-invariant Riemannian metric:

```math
d_R(A,B)=\left\|\log\left(A^{-1/2}BA^{-1/2}\right)\right\|_F
```

Riemannian methods classify them by projecting covariance matrices to the tangent space around a reference mean covariance:

```math
S_i = \log_{\bar{\Sigma}}(\Sigma_i)
```

More explicitly:

```math
S_i = \bar{\Sigma}^{1/2}\log\left(\bar{\Sigma}^{-1/2}\Sigma_i\bar{\Sigma}^{-1/2}\right)\bar{\Sigma}^{1/2}
```

The tangent representation can then be vectorized and passed into a conventional classifier such as LDA. In this project, [src/br_analysis/models.py](src/br_analysis/models.py) uses pyRiemann `Covariances(estimator="oas")`, `TangentSpace(metric="riemann")`, and shrinkage LDA.

This approach is a strong fit for EEG because covariance captures spatial relationships between electrodes and frequency-filtered networks. In the project results, Riemannian covariance features are the strongest or most stable model family.

### LDA

Linear Discriminant Analysis assumes each class is approximately Gaussian with shared covariance. The decision rule is linear:

```math
\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top\Sigma^{-1}\mu_k + \log \pi_k
```

The implementation uses shrinkage to stabilize covariance estimates when samples are limited.

### Deep learning

Deep learning models learn hierarchical representations directly from EEG arrays or graph features. The project uses cross-entropy loss:

```math
L = -\sum_i \sum_k y_{ik}\log p_{ik}
```

Neural models output logits `z`, which are converted to class probabilities by softmax:

```math
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
```

For imbalanced fold-level batches, class-weighted cross-entropy can be used:

```math
L = -\sum_i \alpha_{y_i}\log p_{i,y_i}
```

The main risk is overfitting: the EEG dataset is relatively small, noisy, and subject-specific. Therefore, the project uses early stopping, class weights, dropout, weight decay, standardized inputs, and a two-stage refit strategy in several notebooks.

Weight decay penalizes large weights:

```math
L_{total}=L_{CE}+\lambda ||\theta||_2^2
```

Dropout randomly masks hidden activations during training:

```math
\tilde{h}=m\odot h,\quad m_j\sim Bernoulli(1-p)
```

Early stopping selects the epoch with best validation loss or accuracy. Some notebooks then refit the model on the full training fold for exactly the selected number of epochs, which uses validation information to choose training duration but not to train final weights directly.

### EEGNet

EEGNet is a compact convolutional architecture designed for EEG. In [src/br_analysis/torch_models.py](src/br_analysis/torch_models.py), it applies:

- Temporal convolution to learn frequency-sensitive filters.
- Depthwise spatial convolution to learn channel combinations.
- Separable convolution to combine spatiotemporal features efficiently.
- Pooling and dropout before a linear classifier.

The project evaluates two variants:

- **Agnostic EEGNet**: all 60 channels, broad 1-40 Hz filtering, full epoch.
- **Clustering EEGNet**: data-driven theta/alpha channel clusters and selected time windows from notebook 10.

A temporal convolutional filter computes:

```math
y_{f,c,t}=\sum_{\tau=0}^{K-1} w_{f,\tau}x_{c,t-\tau}
```

Depthwise spatial convolution then learns a spatial filter separately for each temporal feature map:

```math
s_{f,d,t}=\sum_{c=1}^{C} a_{f,d,c}y_{f,c,t}
```

This is why EEGNet is attractive for EEG: it explicitly separates temporal-frequency filtering from spatial channel mixing.

### ST-GNN

The spatio-temporal graph neural network represents electrodes as graph nodes. Edges are based on electrode distances. Each trial is converted into windowed node features, including theta and alpha differential entropy/log-power estimates. The graph convolution uses Chebyshev polynomial filters:

```math
g_\theta *_G x \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L})x
```

where `T_k` are Chebyshev polynomials and `\tilde{L}` is a scaled graph Laplacian. This model is biologically motivated because EEG electrodes have a spatial arrangement and nearby sensors may carry related activity.

The adjacency matrix is based on electrode distance. A common form, also reflected in the project code, is:

```math
A_{ij} =
\begin{cases}
\exp(-d_{ij}^2/\sigma^2), & d_{ij}<r \\
0, & otherwise
\end{cases}
```

The normalized graph Laplacian is:

```math
L=I-D^{-1/2}AD^{-1/2}
```

The graph view matters because EEG is not an unordered vector of channels: electrode geometry encodes useful prior structure.

### Transformers and LoRA

ST-EEGFormer and LaBraM use transformer-style attention. Self-attention computes:

```math
Attention(Q,K,V) = softmax\left(\frac{QK^\top}{\sqrt{d}}\right)V
```

This allows the model to learn interactions across tokens, which may represent channels, temporal patches, or both.

Because pretrained models are large and the dataset is small, the project uses Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning:

```math
W' = W + \Delta W,\quad \Delta W = BA,\quad rank(\Delta W) = r
```

Only small low-rank adapter matrices and classification heads are trained while much of the pretrained backbone remains frozen. This reduces trainable parameters and overfitting risk.

If `W \in \mathbb{R}^{d_{out}\times d_{in}}`, LoRA uses:

```math
A\in\mathbb{R}^{r\times d_{in}},\quad B\in\mathbb{R}^{d_{out}\times r},\quad r \ll \min(d_{in},d_{out})
```

so the number of trainable adapter parameters is:

```math
r(d_{in}+d_{out})
```

instead of:

```math
d_{in}d_{out}
```

This is the main reason LoRA is plausible for a small EEG dataset: it adapts a pretrained representation without updating every backbone parameter.

### Explainability and attribution

The explainability notebooks use saliency and SHAP-style gradient explanations to ask which inputs influence a trained neural model. A simple saliency map is the absolute gradient of the class score with respect to the input:

```math
S_{c,t}=\left|\frac{\partial f_k(X)}{\partial X_{c,t}}\right|
```

Large values indicate that a small perturbation at channel `c`, time `t` would strongly change the class score. This is not the same as proving a neural source is causally involved, but it can show whether a model is relying on plausible time-channel regions.

For perturbation-style attribution, the conceptual Shapley value for feature `j` is:

```math
\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}\left[f(S\cup\{j\})-f(S)\right]
```

Exact Shapley values are too expensive for high-dimensional EEG, so practical SHAP implementations approximate them. In the thesis, these maps should be treated as model-behavior diagnostics, not direct neuroscience proof.

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

The notebooks form a logical workflow. Some notebooks are paired: one version for C vs IP and another for IM vs IP. The paired notebooks are not repeated in full here; the important differences are the task labels, selected subjects, cluster definitions, and results.

### 01_exploració_dades.ipynb

This is the exploratory data analysis notebook and should be treated as the starting point of the thesis methodology.

It loads the Google Drive EEG epochs and examines the behavioral and EEG structure of the dataset. Stored outputs indicate 47 included subjects in the broad exploration and global class counts of approximately:

- Congruent: 8471 trials, 37.1%.
- Incongruent total: 14367 trials, 62.9%.
- Incongruent Pure: 4689 trials, 20.5% of total.
- Incongruent Mixed: 9678 trials.

The notebook studies subject-level dominance:

- IM is much more frequent than IP globally, with an IM ratio around 0.674.
- Left-eye coherent pure percepts are more frequent than right-eye coherent pure percepts, around 0.596 left-eye proportion.
- Red responses are also more frequent than green in the stored dominance summaries.

This matters because a classifier could exploit behavioral or temporal biases if the evaluation design is careless. The notebook therefore motivates class balancing, subject selection, and temporal/grouped validation.

The notebook also computes real Cohen's d estimates for theta at FCz and alpha at Oz, explores subject eligibility, and exports selected-subject tables:

- `subjectes_seleccionats.csv` for C vs IP.
- `subjectes_seleccionats_im_vs_ip.csv` for IM vs IP.

Finally, it reproduces thesis-relevant EEG summaries: ERP checks, theta time courses, alpha suppression, differential time-frequency maps, and fatigue/session-progress analyses. The fatigue section finds only a small error increase over time, not a dramatic collapse of data quality.

### 02_experiments.ipynb

This notebook is a broad hierarchical representational testing framework. It is less a single final model and more an experimental design notebook for asking what type of representation is useful.

It contains 11 experiments:

1. **Preprocessing and normalization ablation**: compares no normalization, trial-wise z-scoring, global fold z-scoring, subject-level scaling, standard scaling, and robust scaling. Stored outputs show only modest differences, with robust scaling slightly best in one C vs IP run.
2. **Representation diagnostics**: tests spatial, temporal, and spectral choices. Outputs suggest C vs IP is more decodable than IM vs IP in several settings, and that the conflict-related time window and broadband/theta-alpha representations are worth exploring.
3. **Classical ML branch**: compares logistic regression, LDA, random forest, XGBoost, and LightGBM. Logistic regression is the strongest classical vector-feature baseline in the stored output.
4. **Classical spatial baselines**: compares CSP, FBCSP, and Riemannian covariance. Riemannian covariance + LDA is best among these in the stored C vs IP output.
5. **Connectivity and network representations**: evaluates theta/alpha phase-locking value features. Stored outputs are close to chance, suggesting simple PLV features are not strongly informative in this setup.
6. **Focused deep learning branch**: evaluates EEGNet variants and deep-learning preprocessing choices.
7. **Pretrained models and transfer learning**: tests ST-EEGFormer and LaBraM-style pretrained backbones with adaptation.
8. **Cohort-level joint search**: uses Optuna-style search to tune shared preprocessing/model configurations.
9. **Deep-learning cohort-level tuning**: tunes model-specific deep learning choices under subject-independent or cohort-level settings.
10. **Controlled contribution ablation**: isolates the contribution of preprocessing, spatial selection, spectral selection, time windows, and model family.
11. **Generalization and control tests**: checks whether models learn condition-relevant structure rather than artifacts or subject identity.

The notebook is important for the thesis because it gives the project a research-design logic: start from simple hypotheses, test representation families, then escalate to more complex models only when justified.

The most up-to-date cached outputs from this notebook add several thesis-relevant points:

- Experiment 8 selects **Riemann + trial-wise z-scoring + all channels + broadband + full epoch** for C vs IP, and **classical ML + robust scaling + ROI + alpha + full epoch + Welch PSD** for IM vs IP.
- Experiment 9's deep-learning tuning reports EEGNet as the strongest tuned neural model among the tested neural/foundation options, with LaBraM and ST-EEGFormer behind it in the cached summary.
- Experiment 10's controlled ablation indicates that band and spatial setting are especially important for C vs IP, while IM vs IP depends more broadly on spatial setting, band, model family, and time scope.
- Experiment 11 includes generalization and subject-identity controls. The subject-identification control is extremely high, which is not surprising for within-cohort held-out trials and reinforces a key warning: EEG models can learn subject fingerprints very easily. This makes subject-aware validation and cautious interpretation essential.

### 03_chance_level_c_ip.ipynb and 07_chance_level_im_ip.ipynb

These notebooks evaluate simple logistic regression decoding using two targeted band/channel features:

- FCz-theta.
- Oz-alpha.

The method is intentionally simple: filter the signal to the selected band, extract mean Hilbert envelope or power-like features over the analysis window, standardize inside the modeling pipeline, and classify with L2 logistic regression. The notebooks run an epoch-grid analysis to see how performance changes as the number of trials increases, and they compare real accuracy to label-permutation chance distributions.

For **C vs IP**, stored outputs show:

- 24 unique subjects.
- Global real mean accuracy around 0.5182 in the epoch grid.
- 22/284 combinations above the 95th percentile of chance.
- FCz-theta: 5/142 combinations above q95, average delta accuracy around 0.0071, maximum accuracy 0.6068.
- Oz-alpha: 17/142 combinations above q95, average delta accuracy around 0.0312, maximum accuracy 0.6437.
- Cohort CV averages around 0.5020 for FCz-theta and 0.5333 for Oz-alpha.

This suggests that single-channel alpha carries more decodable signal than single-channel theta in this simple setup, although the overall effect is modest.

For **IM vs IP**, stored outputs show:

- 29 unique subjects.
- Global real mean accuracy around 0.5020 in the epoch grid.
- 6/324 combinations above q95.
- FCz-theta: 1/162 combinations above q95, average delta accuracy around -0.0014.
- Oz-alpha: 5/162 combinations above q95, average delta accuracy around 0.0061.
- Cohort CV averages around 0.4945 for FCz-theta and 0.5165 for Oz-alpha.

This supports the interpretation that IM vs IP is harder to decode with simple targeted features than C vs IP.

### 04_riemann_c_ip.ipynb and 08_riemann_im_ip.ipynb

These notebooks implement the strongest classical feature family: covariance matrices, Riemannian tangent-space projection, and shrinkage LDA.

Instead of reducing each trial to one band-envelope scalar, the Riemannian pipeline keeps multichannel covariance structure. It filters theta and alpha channel clusters separately, concatenates them as a multi-network feature space, estimates covariance matrices using OAS shrinkage, maps SPD matrices to tangent space, and trains LDA.

For **C vs IP**, stored outputs show:

- 24 subjects.
- Global real mean accuracy around 0.5452 in the epoch grid.
- 27/142 combinations above q95.
- Average delta accuracy around 0.0449.
- Maximum subject/setting accuracy around 0.7178.
- Cohort CV fold summary around 0.56 average in the cross-model comparison.

For **IM vs IP**, stored outputs show:

- 29 subjects.
- Global real mean accuracy around 0.5032.
- 6/162 combinations above q95.
- Average delta accuracy around 0.0203.
- Maximum accuracy around 0.6960.
- Cohort CV average around 0.5175.

This is one of the clearest methodological results: covariance structure across theta and alpha networks is more informative than isolated single-channel features, especially for C vs IP.

### 05_best_config_c_ip.ipynb and 09_best_config_im_ip.ipynb

These notebooks apply the locked best configurations selected by Experiment 8. They do not compare alternatives. They report development CV separately from the reserved final holdout test.

The best-configuration logic is important because it separates **model search** from **final evaluation**. In Experiment 8, many choices are explored jointly:

- normalization,
- spatial channel set,
- frequency band,
- time scope,
- model family,
- model-specific hyperparameters.

The cached Experiment 8 summary currently reports:

```text
C vs IP:
  best family: Riemann
  normalization: trial_zscore
  spatial setting: All
  band: Broadband
  time scope: full
  objective score: 0.569633
  mean accuracy: 0.593605

IM vs IP:
  best family: Classical ML
  normalization: robust_scaler
  spatial setting: ROI
  band: Alpha
  time scope: full
  features: Welch PSD
  objective score: 0.558466
  mean accuracy: 0.580945
```

The same notebook includes a controlled ablation analysis. For C vs IP, replacing the selected band and spatial setting causes the largest drops, suggesting that spectral and spatial choices are not arbitrary. For IM vs IP, replacing the spatial setting, band, model family, and time scope all produces meaningful drops in the cached ablation, suggesting that the best IM vs IP configuration is more dependent on the full combination of preprocessing choices.

This should be written carefully in the thesis: Experiment 8 is a configuration-search result, not a final independent proof. Its value is to guide the locked best-configuration notebooks and to show which preprocessing/model components appear important.

### 06_visualitzacions.ipynb

This notebook is a visualization laboratory. It creates multiple interpretive views rather than a single classifier:

- Searchlight spatiotemporal dynamics.
- Subject selection checks.
- Band-specific searchlight maps.
- Statistical heatmaps.
- Sequential topomaps and GIFs.
- Alpha-vs-theta dominance plots.
- Temporal generalization matrix.
- Latency maps.
- Distance-dependent electrode correlations.
- Classification spectrograms.
- ERP images.
- Riemannian tangent-space PCA projection.
- Connectivity graphs.
- 3D cortical/source-location style views.

For the thesis, this notebook helps convert numeric model performance into interpretable figures. Its value is explanatory: it can help show where, when, and how decoding signal appears, and whether it resembles the hypothesized fronto-medial theta and parieto-occipital alpha mechanisms.

### 10_clustering.ipynb

This notebook performs data-driven spatiotemporal clustering. It identifies channel-time regions where condition differences are statistically structured and then exports chosen clusters to JSON for later focused EEGNet and visualization notebooks.

The exported cluster definitions in the stored output are:

```json
{
  "c_vs_ip": {
    "theta": {
      "channels": ["Fz", "FCz"],
      "tmin": 1.171875,
      "tmax": 1.2421875
    },
    "alpha": {
      "channels": ["FC3", "FC1", "FC2", "C3", "C1", "Cz", "C2", "CP3", "CP1", "CPz", "CP2", "CP4", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
      "tmin": 0.375,
      "tmax": 1.2421875
    }
  },
  "im_vs_ip": {
    "theta": {
      "channels": ["Fz", "FCz", "Cz"],
      "tmin": 1.1953125,
      "tmax": 1.2421875
    },
    "alpha": {
      "channels": ["TP7", "CP5", "CP3", "CP1", "P7", "P5", "P3"],
      "tmin": 0.2578125,
      "tmax": 0.421875
    }
  }
}
```

The notebook also compares paper-replication ROIs with data-driven clusters. In the stored outputs:

- Exact paper-replication C vs IP theta does not survive as a significant cluster in the displayed summary, although its minimum p-value is close (`min p = 0.0410` before the stricter cluster conclusion shown by the notebook).
- Exact paper-replication C vs IP alpha shows a significant window from about 440 ms to 1380 ms (`p = 0.0003`).
- Exact paper-replication IM vs IP theta shows a significant window from about 540 ms to 880 ms (`p = 0.0267`).
- Exact paper-replication IM vs IP alpha does not show a significant cluster in the stored summary.
- Data-driven C vs IP theta shows a significant window from about 937.5 ms to 1382.8 ms (`p = 0.0261`).
- Data-driven C vs IP alpha shows a significant window from about 304.7 ms to 1382.8 ms (`p = 0.0009`).
- Data-driven IM vs IP theta and alpha do not show significant clusters in the stored custom-cluster timecourse summaries.
- Global alpha across all electrodes shows a significant C vs IP window from about 580 ms to 1380 ms (`p = 0.0044`), while global broadband summaries are not significant in the stored output.

This pattern suggests that alpha suppression/engagement is one of the more stable sensor-level signatures for the main C vs IP contrast, while IM vs IP remains less stable and more sensitive to the chosen ROI and time-frequency representation.

### 11_EEGNet_c_ip.ipynb and 17_EEGNet_im_ip.ipynb

These notebooks evaluate EEGNet with two variants:

- **Agnostic**: all 60 channels, broad filtering, full epoch.
- **Clustering**: theta/alpha channels and time windows from notebook 10.

For **C vs IP**, stored outputs show:

- 24 subjects.
- Agnostic EEGNet mean accuracy around 0.534.
- Clustering EEGNet mean accuracy around 0.502-0.505 depending on the summary.
- The paired cohort comparison favors the agnostic model in the stored output (`p = 0.0044` t-test, `p = 0.0035` Wilcoxon).

This is interesting because the data-driven cluster restriction does not improve EEGNet in the cached C vs IP results. A likely interpretation is that the focused cluster is useful for interpretable classical analysis, but the neural network benefits from broader context or suffers from reduced input diversity when too aggressively constrained.

For **IM vs IP**, stored outputs show:

- 29 subjects.
- Both agnostic and clustering EEGNet are close to chance at cohort level.
- Average combined-fold accuracy is around 0.513 for agnostic and 0.511 for clustering.
- The paired statistical comparison does not show a meaningful cohort difference (`p = 0.8874` t-test, `p = 0.9274` Wilcoxon).

The P057 single-subject result can look better for clustering in IM vs IP, but the cohort-level result does not support a general improvement.

### 12_ST-GNN_c_ip.ipynb and 18_ST-GNN_im_ip.ipynb

These notebooks evaluate a spatio-temporal graph neural network. The model builds graph features from all 60 electrodes, using theta/alpha windowed features and a distance-based electrode adjacency matrix. A Chebyshev graph convolution learns spatial structure, and temporal convolution learns changes across time windows.

For C vs IP, the stored P057 detailed run gives an average CV accuracy around 0.505 with a strong class imbalance in recall: high recall for Congruent but low recall for Incongruent Pure. In the cross-model comparison, ST-GNN has mean accuracy around 0.515.

For IM vs IP, the stored output shown in the notebook appears to reuse labels in the printed classification report, so it should be treated cautiously. The cross-model IM vs IP comparison did not find cached ST-GNN predictions at the time of the stored output, so the most reliable cohort-level IM vs IP comparison currently comes from LR and Riemannian models.

### 13_ST-EEGFormer_c_ip.ipynb and 19_ST-EEGFormer_im_ip.ipynb

ST-EEGFormer is a transformer-style architecture adapted to EEG. The project aligns the 60 recorded channels to the larger channel layout expected by the pretrained model and uses LoRA adapters for fine-tuning.

For C vs IP, stored outputs show:

- Fold accuracies around 0.50-0.54.
- Average CV accuracy around 0.5179.
- Subject P057 has a better single-subject classification report, around 0.61 accuracy.

This suggests that pretrained transformer features may help some subjects but did not beat Riemannian or EEGNet agnostic methods at the cohort level in the cached results.

For IM vs IP, the current stored output is sparse: it shows that cached ST-EEGFormer cohort metrics are being loaded, but the cross-model IM vs IP notebook still does not include ST-EEGFormer because the expected prediction cache is not present. The thesis should therefore avoid treating ST-EEGFormer IM vs IP as a final benchmark until the notebook is rerun and its predictions are exported consistently.

### 14_LaBraM_c_ip.ipynb and 20_LaBraM_im_ip.ipynb

LaBraM is a large pretrained EEG foundation model. The project pads or adapts the trial representation, loads the pretrained checkpoint, and fine-tunes with LoRA-style adapters and a classifier head.

For C vs IP, stored outputs show:

- Average CV accuracy around 0.4965.
- Subject P057 accuracy around 0.54.
- In the cross-model ranking, LaBraM is last among the included C vs IP models.

This does not mean LaBraM is generally bad; it means that in this small, specific, onset-BR decoding setting, the chosen adaptation and data volume did not produce a clear benefit over classical methods.

For IM vs IP, the current LaBraM notebook structure is present but the stored output does not provide enough cohort-level evidence to include it in the final benchmark. The IM vs IP cross-model notebook still does not include LaBraM cached predictions.

### 15_cross_model_comparison_c_ip.ipynb and 21_cross_model_comparison_im_ip.ipynb

These are the main benchmark notebooks.

For **C vs IP**, notebook 15 merges 24 subjects and 8 model configurations:

```text
Riemann:          mean accuracy 0.5659, std 0.0618
LR_alpha:         mean accuracy 0.5447, std 0.0452
EEGNet_Agnostic:  mean accuracy 0.5338, std 0.0380
ST_EEGFormer:     mean accuracy 0.5178, std 0.0411
ST_GNN:           mean accuracy 0.5150, std 0.0457
LR_theta:         mean accuracy 0.5126, std 0.0433
EEGNet_Clustering:mean accuracy 0.5046, std 0.0469
LaBraM:           mean accuracy 0.4966, std 0.0300
```

The ranking is the strongest quantitative summary currently in the repository. It supports three thesis-level conclusions:

1. The C vs IP contrast contains weak but measurable information.
2. Riemannian covariance features are the most effective model family in the current implementation.
3. More complex deep/foundation models do not automatically improve performance on small EEG datasets.

For **IM vs IP**, notebook 21 currently merges 29 subjects and 3 model configurations because the expected deep-model prediction caches were not found:

```text
Riemann:  mean accuracy 0.5322, std 0.0466
LR_alpha: mean accuracy 0.5083, std 0.0516
LR_theta: mean accuracy 0.5037, std 0.0401
```

The IM vs IP ranking is therefore more limited. It suggests that Riemannian covariance again performs best, but the effect is smaller than for C vs IP.

### 16_explainability_c_ip.ipynb and 22_explainability_im_ip.ipynb

These notebooks centralize interpretability and explainability for the deep models. They are designed to compare saliency mapping and SHAP-style gradient explanations across models.

Their thesis role is to connect model predictions back to neuroscience:

- Do important time windows overlap with stimulus presentation and conflict-resolution intervals?
- Do important channels overlap with fronto-medial theta and parieto-occipital alpha regions?
- Are deep models using plausible EEG structure or unstable artifacts?

The stored outputs in these notebooks are sparse, so they should be treated more as interpretability infrastructure than as final result notebooks unless rerun with the final saved models.

## Main results and interpretation

### Behavioral and dataset structure

The dataset is not balanced naturally. IM trials are much more frequent than IP trials, and subject-level perceptual biases are common. The project handles this through subject selection and random undersampling in the decoding tasks. This is not just a preprocessing detail: it is central to the validity of the thesis, because models could otherwise learn class prevalence or temporal drift rather than neural signatures of conflict.

The selected cohorts used by the main decoding notebooks are:

- 24 subjects for C vs IP.
- 29 subjects for IM vs IP.

### C vs IP is the more robust contrast

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

### Subject identity is a strong control signal

Experiment 11's subject-identification control reaches extremely high accuracy in the cached output. This does not mean the condition classifiers are invalid, but it is a warning that EEG contains stable person-specific structure. For thesis interpretation, this means:

- within-subject decoding answers whether condition information is present for a subject;
- cohort decoding can mix condition information with subject-level structure if validation is not careful;
- LOSO and cross-condition tests are stricter tests of generalization;
- subject identification should be reported as a control, not as a target result.

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

1. **Introduction**: binocular rivalry, perceptual conflict, EEG decoding, motivation for ML comparison.
2. **Background**: EEG, oscillations, theta/alpha, binocular rivalry, conflict monitoring, ML for EEG, validation risks.
3. **Research questions and objectives**: C vs IP and IM vs IP, model comparison, interpretability.
4. **Dataset and preprocessing**: onset rivalry protocol, event codes, epochs, subject selection, class imbalance, Google Drive data organization.
5. **Methods**: feature extraction, logistic regression, Riemannian covariance, EEGNet, ST-GNN, transformers, LaBraM, cross-validation, permutation testing, cluster-based statistics.
6. **Experiments**: notebook pipeline, ablations, clustering, model-specific training, cross-model benchmark.
7. **Results**: behavioral EDA, C vs IP decoding, IM vs IP decoding, model ranking, interpretability.
8. **Discussion**: what the results imply for conflict monitoring, why classical methods worked best, why deep models did not dominate, limitations.
9. **Conclusions and future work**: final answer to the objectives, improvements for reruns, better validation, richer features, subject-transfer analysis.

## Conclusions

This project investigates whether EEG signals recorded during onset binocular rivalry contain decodable information about perceptual conflict and conflict resolution. The most important contrast is C vs IP, where both conditions lead to pure percept reports but differ in whether physical interocular conflict was present.

The current evidence supports a cautious positive answer. The signal is weak, but not random: Riemannian covariance classification reaches the best C vs IP performance in the stored cross-model benchmark, and alpha/theta cluster analyses identify biologically plausible time-channel regions. The IM vs IP contrast is less robust and remains more exploratory.

The methodological conclusion is also clear. For this dataset, carefully designed classical EEG methods outperform or match more complex deep learning and pretrained models. Riemannian covariance features are especially valuable because they preserve distributed spatial structure while remaining appropriate for small EEG datasets. Deep models remain useful as exploratory tools and for interpretability comparisons, but they require more data, careful tuning, and stricter cache/reproducibility checks before being treated as final evidence.

The thesis should therefore frame the project as a rigorous EEG decoding study rather than a high-accuracy prediction challenge. Its contribution is the systematic comparison of interpretable and modern ML approaches for binocular-rivalry EEG, showing that conflict-related perceptual information is present but subtle, distributed, and highly sensitive to validation design.
