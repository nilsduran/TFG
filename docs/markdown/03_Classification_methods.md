### Journal of Neural Engineering

###### TOPICAL REVIEW

## A review of classification algorithms for EEG-

## based brain–computer interfaces: a 10 year

## update

To cite this article: F Lotte et al 2018 J. Neural Eng. 15 031005

View the article online for updates and enhancements.

### You may also like

```
Defining and quantifying users’ mental
imagery-based BCI skills: a first step
Fabien Lotte and Camille Jeunet
```
-

```
Expanding the (kaleido)scope: exploring
current literature trends for translating
electroencephalography (EEG) based
brain–computer interfaces for motor
rehabilitation in children
E Kinney-Lang, B Auyeung and J
Escudero
```
-

```
A study of the existing problems of
estimating the information transfer rate in
online brain–computer interfaces
Peng Yuan, Xiaorong Gao, Brendan
Allison et al.
```
-

```
This content was downloaded from IP address 84.89.157.30 on 30/10/2023 at 16:
```

```
1 © 2018 IOP Publishing Ltd Printed in the UK
```
```
Journal of Neural Engineering
```
F Lotte _et al_

Review of classification algorithms for EEG-based BCI

Printed in the UK

031005

JNEIEZ

© 2018 IOP Publishing Ltd

15

J. Neural Eng.

JNE

1741-

10.1088/1741-2552/aab2f

##### 3

Journal of Neural Engineering

#### F Lotte^1 ,^2 , L Bougrain^3 ,^4 , A Cichocki^2 ,^5 ,^10 , M Clerc^6 , M Congedo^7 ,

#### A Rakotomamonjy^8 and F Yger^9

(^1) Inria, LaBRI (CNRS/Univ. Bordeaux /INP), Talence, France
(^2) RIKEN Brain Science Insitute, Wakoshi, Japan
(^3) Univ. Lorraine, Nancy, France
(^4) Inria Nancy Grand-Est / LORIA, Nancy, France
(^5) Skolkowo Institute of Science and Technology, Moscow, Russia
(^6) Inria, Université Côte d’Azur, France
(^7) Univ. Grenoble Alpes, CNRS, Grenoble INP, GIPSA-lab, Grenoble, France
(^8) Univ. Rouen / LITIS, Rouen, France
(^9) Univ. Paris-Dauphine, PSL Research Univ. / CNRS, LAMSADE, Paris, France
(^10) Nicolaus Copernicus University, Torun, Poland
E-mail: fabien.lotte@inria.fr
Received 21 November 2017, revised 23 January 2018
Accepted for publication 28 February 2018
Published 16 April 2018
**Abstract**

##### Objective. Most current electroencephalography (EEG)-based brain–computer interfaces

##### (BCIs) are based on machine learning algorithms. There is a large diversity of classifier types

##### that are used in this field, as described in our 2007 review paper. Now, approximately ten years

##### after this review publication, many new algorithms have been developed and tested to classify

##### EEG signals in BCIs. The time is therefore ripe for an updated review of EEG classification

##### algorithms for BCIs. Approach. We surveyed the BCI and machine learning literature from

##### 2007 to 2017 to identify the new classification approaches that have been investigated to

##### design BCIs. We synthesize these studies in order to present such algorithms, to report how

##### they were used for BCIs, what were the outcomes, and to identify their pros and cons. Main

##### results. We found that the recently designed classification algorithms for EEG-based BCIs can

##### be divided into four main categories: adaptive classifiers, matrix and tensor classifiers, transfer

##### learning and deep learning, plus a few other miscellaneous classifiers. Among these, adaptive

##### classifiers were demonstrated to be generally superior to static ones, even with unsupervised

##### adaptation. Transfer learning can also prove useful although the benefits of transfer learning

##### remain unpredictable. Riemannian geometry-based methods have reached state-of-the-art

##### performances on multiple BCI problems and deserve to be explored more thoroughly, along

##### with tensor-based methods. Shrinkage linear discriminant analysis and random forests also

##### appear particularly useful for small training samples settings. On the other hand, deep learning

##### methods have not yet shown convincing improvement over state-of-the-art BCI methods.

##### Significance. This paper provides a comprehensive overview of the modern classification

##### algorithms used in EEG-based BCIs, presents the principles of these methods and guidelines

##### on when and how to use them. It also identifies a number of challenges to further advance

##### EEG classification in BCI.

# A review of classification algorithms

# for EEG-based brain–computer interfaces:

# a 10 year update

## Topical Review

```
IOP
```
2018

```
1741-2552/18/031005+28$33.
```
```
J. Neural Eng. 15 (2018) 031005 (28pp) https://doi.org/10.1088/1741-2552/aab2f
```

##### Keywords: brain–computer interfaces, electroencephalography, classification, spatial filtering,

##### deep learning, Riemannian geometry, transfer learning

##### (Some figures may appear in colour only in the online journal)

##### 1. Introduction

##### A brain–computer interface (BCI) can be defined as a system

##### that translates the brain activity patterns of a user into mes-

##### sages or commands for an interactive application, this activity

##### being measured and processed by the system [44, 139, 229].

##### A BCI user’s brain activity is typically measured using elec-

##### troencephalography (EEG). For instance, a BCI can enable a

##### user to move a cursor to the left or to the right of a computer

##### screen by imagining left or right hand movements, respec-

##### tively [230]. As they make computer control possible without

##### any physical activity, EEG-based BCIs promise to revolution-

##### ize many applications areas, notably to enable severely motor-

##### impaired users to control assistive technologies, e.g. text input

##### systems or wheelchairs [181], as rehabilitation devices for

##### stroke patients [8], as new gaming input devices [52], or to

##### design adaptive human–computer interfaces that can react to

##### the user’s mental states [237], to name a few [45, 216].

##### In order to use a BCI, two phases are generally required:

##### (1) an offline training phase during which the system is cali-

##### brated and (2) the operational online phase in which the sys-

##### tem can recognize brain activity patterns and translate them

##### into commands for a computer [136]. An online BCI system

##### is a closed-loop, starting with the user producing a specific

##### EEG pattern (e.g. using motor imagery) and these EEG sig-

##### nals being measured. Then, EEG signals are typically pre-

##### processed using various spatial and spectral filters [23], and

##### features are extracted from these signals in order to represent

##### them in a compact form [140]. Finally, these EEG features are

##### classified [141] before being translated into a command for an

##### application [45] and before feedback is provided to users to

##### inform them whether a specific mental command was recog-

##### nized or not [170].

##### Although much effort is currently under way towards

##### calibration-free modes of operation, an off-line calibration is

##### currently used and is necessary in most BCIs to obtain a reli-

##### able system. In this stage, the classification algorithm is cali-

##### brated and the optimal features from multiple EEG channels

##### are selected. For this calibration, a training data set needs to

##### be pre-recorded from the user. EEG signals are highly user-

##### specific, and as such, most current BCI systems are calibrated

##### specifically for each user. This training data set contains EEG

##### signals recorded while the user performed each mental task of

##### interest several times, according to given instructions.

##### There are various key elements in the BCI closed-loop,

##### one being the classification algorithms a.k.a classifiers used

##### to recognize the users’ EEG patterns based on EEG features.

##### There was, and still is, a large diversity of classifier types

##### that are used and have been explored to design BCIs, as pre-

##### sented in our 2007 review of classifiers for EEG-based BCIs

##### [141]. Now, approximately ten years after this initial review

##### was published, many new algorithms have been designed and

##### explored in order to classify EEG signals in BCI, and BCIs are

##### more popular than ever. We therefore believe that the time is

##### ripe to update this review of EEG classifiers. Consequently, in

##### this paper, we survey the literature on BCI and machine learn-

##### ing from 2007 to 2017 in order to identify which new EEG

##### classification algorithms have been investigated to design

##### BCI, and which appear to be the most efficient^11. Note that we

##### also include in the present review machine learning methods

##### for EEG feature extraction, notably to optimize spatial filters,

##### which have become a key component of BCI classification

##### approaches. We synthesize these readings in order to present

##### these algorithms, to report how they were used for BCIs and

##### what were the outcomes. We also identify their pros and cons

##### in order to provide guidelines regarding how and when to use

##### a specific classification method, and propose some challenges

##### that must be solved to enable further progress in EEG signal

##### classification.

##### This paper is organized as follows. Section 2 briefly pre-

##### sents the typically used EEG feature extraction and selection

##### techniques, as these features are usually the input to classi-

##### fiers. It also summarizes the classifier performance evaluation

##### metrics. Then, section 3.1 provides a summary of the classi-

##### fiers that were used for EEG-based BCIs up to 2007, many of

##### which are still in use today, as well as the challenges faced by

##### current EEG classification methods. Section 4 describes the

##### core of the paper, as it reviews the classification algorithms for

##### BCI that have been explored since 2007 to address these vari-

##### ous challenges. These algorithms are discussed in section 5,

##### where we also propose guidelines on how and when to use

##### them, and identify some remaining challenges. Finally, sec-

##### tion 6 concludes the paper.

##### 2. Feature extraction and selection,

##### and performance measures in brief

##### The present paper is dedicated to classification methods for

##### BCI. However, most pattern recognition/machine learning

##### pipelines, and BCIs are no exception, not only use a classi-

##### fier, but also apply feature extraction/selection techniques to

##### represent EEG signals in a compact and relevant manner. In

##### particular for BCI, EEG signals are typically filtered both in

##### the time domain (band-pass filter), and spatial domain (spatial

##### filter) before features are extracted from the resulting signals.

##### The best subsets of features are then identified using feature

(^11) This updated review describes more advanced classification concepts and
algorithms than the ones presented in the initial review in [141]. We thus
advise our readers new to the EEG classification field to start by reading
[141], as that paper is more accessible, and the concepts it presented will not
be explained again in the current manuscript.
J. Neural Eng. **15** ( 2018 ) 031005


##### selection algorithms, and these features are used to train a

##### classifier. This process is illustrated in figure 1. In this chapter,

##### we briefly discuss which features are typically used in BCI,

##### how to select the most relevant features amongst these and

##### how to evaluate the resulting pattern recognition pipeline.

2.1. Feature extraction

##### While there are many ways in which EEG signals can be rep-

##### resented (e.g. [16, 136, 155]), the two most common types

##### of features used to represent EEG signals are frequency band

##### power features and time point features.

##### Band power features represent the power (energy) of EEG

##### signals for a given frequency band in a given channel, averaged

##### over a given time window (typically 1 second for many BCI

##### paradigms). Band power features can be computed in various

##### ways [28, 87], and are extensively used for BCIs exploiting

##### oscillatory activity, i.e. changes in EEG rhythm amplitudes.

##### As such, band power features are the gold standard features

##### for BCI based on motor and mental imagery for many passive

##### BCI aiming at decoding mental states such as mental work-

##### load or emotions, or for steady state visual evoked potential

##### (SSVEP)-based BCIs.

##### Time point features are a concatenation of EEG samples

##### from all channels. Typically, such features are extracted after

##### some pre-processing, notably band-pass or low-pass filter-

##### ing and down-sampling. They are the typical features used to

##### classify Event Related Potentials (ERP), which are temporal

##### variations in EEG signals’ amplitudes time-locked to a given

##### event/stimulus [22, 136]. These are the features used in most

##### P300-based BCI.

##### Both types of features benefit from being extracted after

##### spatial filtering [22, 136, 185, 188]. Spatial filtering consists

##### of combining the original sensor signals, usually linearly,

##### which can result in a signal with a higher signal-to-noise ratio

##### than that of individual sensors. Spatial filtering can be data

##### independent, e.g. based on physical consideration regarding

##### how EEG signals travel through the skin and skull, lead-

##### ing to spatial filters such as the well-known Laplacian filter

##### [159] or inverse solution based spatial filtering [18, 101, 124,

##### 173]. Spatial filters can also be obtained in a data-driven and

##### unsupervised manner with methods such as principal comp-

##### onent analysis (PCA) or independent component analysis

##### (ICA) [98]. Finally, spatial filters can be obtained in a data-

##### driven manner, with supervised learning, which is currently

##### one of the most popular approaches. Supervised spatial fil-

##### ters include the well-known common spatial patterns (CSP)

##### [23, 185], dedicated to band-power features and oscillatory

##### activity BCI, and spatial filters such as xDAWN [188] or

##### Fisher spatial filters [92] for ERP classification based on time

##### point features. Owing to the good classification performances

##### obtained by such supervised spatial filters in practice, many

##### variants of such algorithms have been developed that are more

##### robust to noise or non-stationary signals, using regularization

##### approaches, robust data averaging, and/or new divergence

##### measures, (e.g. [143, 187, 194, 211, 233]). Similarly, exten-

##### sions of these approaches have been proposed to optimize

##### spectral and spatial filters simultaneously (e.g. the popular fil-

##### ter bank CSP (FBCSP) method [7] and others [61, 88, 161]).

##### Finally, some approaches have combined both physically-

##### driven spatial filters based on inverse models with data-driven

##### spatial filters (e.g. [49, 148]).

##### While spatial filtering followed by either band power or

##### time points feature extraction are by far the most common fea-

##### tures used in current EEG-based BCIs, it should be mentioned

##### that other feature types have been explored and used. Firstly,

##### an increasingly used type is connectivity features. Such fea-

##### tures measure the correlation or synchronization between

##### signals from different sensors and/or frequency bands. This

##### can be measured using features such as spectral coherence,

##### phase locking values or directed transfer functions, among

##### many others [31, 79, 110, 167, 225, 240]. Researchers have

##### also explored various EEG signal complexity measures or

##### higher order statistics as features of EEG signals (e.g. [11,

**Figure 1.** Typical classification process in EEG-based BCI systems. The oblique arrow denotes algorithms that can be or have to be
optimized from data. A training phase is typically necessary to identify the best filters and features and to train the classifier. The resulting
filters, features and classifier are then used online to operate the BCI.

J. Neural Eng. **15** ( 2018 ) 031005


##### 29, 135, 248]). Finally, rather than using vectors of features,

##### recent research has also explored how to represent EEG sig-

##### nals by covariance matrices or by tensors (i.e. arrays and

##### multi-way arrays, with two or more dimensions), and how to

##### classify these matrices or tensors directly [38, 47, 232]. Such

##### approaches are discussed in section 4.2. It should be men-

##### tioned that when using matrix or tensor decompositions, the

##### resulting features are linear combinations of various sensors’

##### data, time points or frequencies (among others). As such they

##### may not have an obvious physical/physiological interpreta-

##### tion, but nonetheless prove useful for BCI design.

##### Finally, it is interesting to note that several BCI studies

##### have reported that combining various types of features, e.g.

##### time points with band powers or band powers with connectiv-

##### ity features, generally leads to higher classification accuracies

##### as compared to using a single feature type (e.g. [29, 60, 70,

##### 93, 166, 191]). Combining multiple feature types typically

##### increases dimensionality; hence it requires the selection of the

##### most relevant features to avoid the curse-of-dimensionality.

##### Methods to reduce dimensionality are described in the follow-

##### ing section.

2.2. Feature selection

##### A feature selection step can be applied after the feature extrac-

##### tion step to select a subset of features with various potential

##### benefits [82]. Firstly, among the various features that one may

##### extract from EEG signals, some may be redundant or may not

##### be related to the mental states targeted by the BCI. Secondly,

##### the number of parameters that the classifier has to optimize is

##### positively correlated with the number of features. Reducing

##### the number of features thus leads to fewer parameters to be

##### optimized by the classifier. It also reduces possible overtrain-

##### ing effects and can thus improve performance, especially

##### if the number of training samples is small. Thirdly, from a

##### knowledge extraction point of view, if only a few features are

##### selected and/or ranked, it is easier to observe which features

##### are actually related to the targeted mental states. Fourthly, a

##### model with fewer features and consequently fewer param-

##### eters can produce faster predictions for a new sample, as it

##### should be computationally more efficient. Fifthly, collection

##### and storage of data will be reduced. Three feature selection

##### approaches have been identified [106]: the filter, wrapper and

##### embedded approaches. Many alternative methods have been

##### proposed for each approach.

##### Filter methods rely on measures of relationship between

##### each feature and the target class, independently of the clas-

##### sifier to be used. The coefficient of determination, which is

##### the square of the estimation of the Pearson correlation coef-

##### ficient, can be used as a feature ranking criterion [85]. The

##### coefficient of determination can also be used for a two-class

##### problem, labelling classes as −1 or +1. The correlation coef-

##### ficient can only detect linear dependencies between features

##### and classes though. To exploit non-linear relationships, a

##### simple solution is to apply non-linear pre-processing, such as

##### taking the square or the log of the features. Ranking criteria

##### based on information theory can also be used e.g. the mutual

##### information between each feature and the target variable [82,

##### 180]. Many filter feature selection approaches require esti-

##### mations of the probability densities and the joint density of

##### the feature and class label from the data. One solution is to

##### discretize the features and class labels. Another solution is

##### to approximate their densities with a non-parametric method

##### such as Parzen windows [179]. If the densities are estimated

##### by a normal distribution, the result obtained by the mutual

##### information will be similar to the one obtained by the corre-

##### lation coefficient. Filter approaches have a linear complexity

##### with respect to the number of features. However, this may lead

##### to a selection of redundant features [106].

##### Wrapper and embedded approaches solve this problem at

##### the cost of a longer computation time. These approaches use

##### a classifier to obtain a subset of features. Wrapper methods

##### select a subset of features, present it as input to a classifier for

##### training, observe the resulting performance and stop the search

##### according to a stopping criterion or propose a new subset if

##### the criterion is not satisfied. Embedded methods integrate the

##### features selection and the evaluation in a unique process, e.g.

##### in a decision tree [27, 184] or a multilayer perceptron with

##### optimal cell damage [37].

##### Feature selection has provided important improvements in

##### BCI, e.g. the stepwise linear discriminant analysis (embed-

##### ded method) for P300-BCI [111] and frequency bands selec-

##### tion for motor imagery using maximal mutual information

##### (filtering methods) [7]. Let us also mention the support vec-

##### tor machine for channel selection [115], linear regressor for

##### knowledge extraction [123], genetic algorithms for spectral

##### feature selection [50] and P300-based feature selection [201],

##### or evolutionary algorithms for feature selection based on

##### multiresolution analysis [176] (all being wrapper methods).

##### Indeed, metaheuristic techniques (also including ant colony,

##### swarm search, tabu search and simulated annealing) [152] are

##### becoming more and more frequently used for feature selection

##### in BCI [174] in order to avoid the curse-of-dimensionality.

##### Other popular methods used in EEG-based BCIs notably

##### include filter methods such as maximum relevance minimum

##### redundancy (mRMR) feature selection [166, 180] or R^2 fea-

##### ture selection [169, 217]. It should be mentioned that five

##### feature selection methods, namely information gain ranking,

##### correlation-based feature selection, Relief (an instance-based

##### feature ranking method for multiclass problems), consistency-

##### based feature selection and 1R Ranking (one-rule classifica-

##### tion) have been evaluated on the BCI competition III data sets

##### [107]. Amongst ten classifiers, the top three feature selection

##### methods were correlation-based feature selection, information

##### gain and 1R ranking, respectively.

```
2.3. Performance measures
```
##### To evaluate BCI performance, one must bear in mind that

##### different components of the BCI loop are at stake [212].

##### Regarding the classifier alone, the most basic performance

##### measure is the classification accuracy. This is valid only if the

##### classes are balanced [66], i.e. with the same number of samples

##### per class and if the classifier is unbiased, i.e. it has the same

J. Neural Eng. **15** ( 2018 ) 031005


##### performance for each class [199]. If these conditions are not

##### met, the Kappa metric or the confusion matrix are more infor-

##### mative performance measures [66]. The sensitivity-specificity

##### pair, or precision, can be computed from the confusion matrix.

##### When the classification depends on a continuous parameter

##### (e.g. a threshold), the receiver operating characteristic (ROC)

##### curve, and the area under the curve (AUC) are often used.

##### Classifier performance is generally computed offline on

##### pre-recorded data, using a hold-out strategy: some datasets

##### are set aside to be used for the evaluation, and are not part of

##### the training dataset. However, some authors also report cross-

##### validation measures estimated on training data, which may

##### over-rate the performance.

##### The contribution of classifier performance to overall BCI

##### performance strongly depends on the orchestration of the BCI

##### subcomponents. This orchestration is highly variable given the

##### variety of BCI systems (co-adaptive, hybrid, passive, self- or

##### system- paced). The reader is referred to [212] for a compre-

##### hensive review of evaluation strategies in such BCI contexts.

##### 3. Past methods and current challenges

3.1. A brief overview of methods used ten years ago

##### In our original review of classification algorithms for EEG-

##### based BCIs published ten years ago, we identified five main

##### families of classifiers that had been explored: linear classi-

##### fiers, neural networks, non-linear Bayesian classifiers, nearest

##### neighbour classifiers and classifier combinations [141].

##### Linear classifiers gather discriminant classifiers that use

##### linear decision boundaries between the feature vectors of each

##### class. They include linear discriminant analysis (LDA), regu-

##### larized LDA and support vector machines (SVMs). Both LDA

##### and SVM were, and still are, the most popular types of clas-

##### sifiers for EEG based-BCIs, particularly for online and real-

##### time BCIs. The previous review highlighted that in terms of

##### performances, SVM often outperformed other classifiers.

##### Neural networks (NN) are assemblies of artificial neurons,

##### arranged in layers, which can be used to approximate any non-

##### linear decision boundary. The most common type of NN used

##### for BCI at that time was the multi-layer perceptron (MLP),

##### typically employing only one or two hidden layers. Other NN

##### types were explored more marginally, such as the Gaussian

##### classifier NN or learning vector quantization (LVQ) NN.

##### Non-linear Bayesian classifiers are classifiers modeling the

##### probability distributions of each class and use Bayes’ rule to

##### select the class to assign to the current feature vector. Such

##### classifiers notably include Bayes quadratic classifiers and hid-

##### den Markov models (HMMs).

##### Nearest neighbour classifiers assign a class to the cur-

##### rent feature vector according to its nearest neighbours. Such

##### neighbours could be training feature vectors or class proto-

##### types. Such classifiers include the k-nearest neighbour (kNN)

##### algorithm or Mahalanobis distance classifiers.

##### Finally, classifier combinations are algorithms combin-

##### ing multiple classifiers, either by combining their outputs

##### and/or by training them in ways that maximize their comple-

##### mentarity. Classifier combinations used for BCI at the time

##### included boosting, voting or stacking combination algorithms.

##### Classifier combination appeared to be amongst the best per-

##### forming classifiers for EEG based BCIs, at least in offline

##### evaluations.

```
3.2. Challenges faced by current EEG signal classification
methods
```
##### Ten years ago, most classifiers explored for BCI were rather

##### standard classifiers used in multiple machine learning

##### problems. Since then, research efforts have focused on iden-

##### tifying and designing classification methods dedicated to the

##### specificities of EEG-based BCIs. In particular, the main chal-

##### lenges faced by classification methods for BCI are the low

##### signal-to-noise ratio of EEG signals [172, 228], their non-

##### stationarity over time, within or between users, where same-

##### user EEG signals varying between or even within runs [56, 80,

##### 109, 145, 164, 202], the limited amount of training data that is

##### generally available to calibrate the classifiers [108, 137], and

##### the overall low reliability and performance of current BCIs

##### [109, 138, 139, 229].

##### Therefore, most of the algorithms studied these past ten

##### years aimed at addressing one or more of these challenges.

##### More precisely, adaptive classifiers whose parameters are

##### incrementally updated online were developed to deal with

##### EEG non-stationarity in order to track changes in EEG prop-

##### erties over time. Adaptive classifiers can also be used to deal

##### with limited training data by learning online, thus requir-

##### ing fewer offline training data. Transfer learning techniques

##### aim at transferring features or classifiers from one domain,

##### e.g. BCI subjects or sessions, to another domain, e.g. other

##### subjects or other sessions from the same subject. As such

##### they also aim at addressing within or between-subjects non-

##### stationarity and limited training data by complementing the

##### few training data available with data transferred from other

##### domains. Finally in order to compensate for the low EEG

##### signal-to-noise ratio and the poor reliability of current BCIs,

##### new methods were explored to process and classify signals in

##### a single step by merging feature extraction, feature selection

##### and classification. This was achieved by using matrix (notably

##### Riemannian methods) and tensor classifiers as well as deep

##### learning. Additional methods explored were targeted specifi-

##### cally at learning from limited amount of data and at dealing

##### with multiple class problems. We describe these new families

##### of methods in the following.

##### 4. New EEG classification methods since 2007

```
4.1. Adaptive classifiers
```
##### 4.1.1. Principles. Adaptive classifiers are classifiers whose

##### parameters, e.g. the weights attributed to each feature in a lin-

##### ear discriminant hyperplane, are incrementally re-estimated

##### and updated over time as new EEG data become available

##### [200, 202]. This enables the classifier to track possibly chang-

##### ing feature distribution, and thus to remain effective even with

##### non-stationary signals such as an EEG. Adaptive classifiers

##### for BCI were first proposed in the mid-2000s, e.g. in [30, 72,

J. Neural Eng. **15** ( 2018 ) 031005


##### 163, 202, 209], and were shown to be promising in offline

##### analysis. Since then, more advanced adaptation techniques

##### have been proposed and tested, including online experiments.

##### Adaptive classifiers can employ both supervised and

##### unsupervised adaptation, i.e. with or without knowledge of

##### the true class labels of the incoming data, respectively. With

##### supervised adaptation, the true class labels of the incoming

##### EEG signals is known and the classifier is retrained on the

##### available training data augmented with these new, labelled

##### incoming data, or is updated based on this new data only [200,

##### 202]. Supervised BCI adaptation requires guided user train-

##### ing, for which the users’ commands are imposed and thus

##### the corresponding EEG class labels are known. Supervised

##### adaptation is not possible with free BCI use, as the incoming

##### EEG data true label is unknown. With unsupervised adap-

##### tation, the label of the incoming EEG data is unknown. As

##### such, unsupervised adaptation is based on an estimation of

##### the data class labels for retraining/updating, as discussed

##### in [104], or is based on class-unspecific adaptation, e.g. the

##### general all classesEEG data mean [24, 219] or a covariance

##### matrix [238] is updated in the classifier model. A third type

##### of adaptation, in between supervised and unsupervised meth-

##### ods, has also been explored: semi-supervised adaptation [121,

##### 122]. Semi-supervised adaptation consists of using both ini-

##### tial labelled data and incoming unlabelled data to adapt the

##### classifier. For BCI, semi-supervised adaptation is typically

##### performed by (1) initially training a supervised classifier on

##### available labelled training data, then (2) by estimating the

##### labels of incoming unlabelled data with this classifier, and

##### (3) by adapting/retraining the classifier using these initially

##### unlabelled data assigned to their estimated labels combined

##### with the known available labelled training data. This process

##### is repeated as new batches of unlabelled incoming EEG data

##### become available.

##### 4.1.2. State-of-the-art. So far, the majority of the work on

##### adaptive classifiers for BCI has been based on supervised

##### adaptation. Multiple adaptive classifiers were explored offline,

##### such as LDA or quadratic discriminant analysis (QDA) [200]

##### for motor imagery-based BCI. An adaptive LDA was also

##### proposed based on Kalman filtering to track the distribution

##### of each class [96]. In order to deal with possibly imperfect

##### labels in supervised adaptation, [236] proposed and evalu-

##### ated offline an adaptive Bayesian classifier based on sequen-

##### tial Monte Carlo sampling that explicitly models uncertainty

##### in the observed labels. For ERP-based BCI, [227] explored

##### an offline adaptive support vector machine (SVM), adaptive

##### LDA, a stochastic gradient-based adaptive linear classifier,

##### and online passive-aggressive (PA) algorithms. interestingly,

##### McFarland and colleagues demonstrated in offline analysis

##### of EEG data over multiple sessions that continuously retrain-

##### ing the weights of linear classifiers in a supervised manner

##### improved the performance of sensori-motor rhythms (SMR)-

##### based BCI, but not of the P300-based BCI speller [160].

##### However, results presented in [197] suggested that continu-

##### ous adaption was beneficial for the asynchronous P300-BCI

##### speller, and [227] suggested the same for passive BCI based

##### on the P300.

##### Online, still using supervised adaptation, both adaptive

##### LDA and QDA have been explored successfully in [222]. In

##### [86], an adaptive probabilistic neural network was also used

##### for online adaptation with a motor imagery-BCI. Such a clas-

##### sifier models the feature distributions of each class in non-

##### parametric fashion, and updates them as new trials become

##### available. Classifier ensembles were also explored to create

##### adaptive classifiers. In [119], a dynamic ensemble of five

##### SVM classifiers was created by training a new SVM for each

##### batch of new incoming labelled EEG trials, adding it to the

##### ensemble and removing the oldest SVM. Classification was

##### performed using a weighted sum of each SVM output. This

##### approach was shown online to be superior to a static classifier.

##### Regarding supervised adaptation, it should be mentioned

##### that adaptive spatial filters were also proposed, notably sev-

##### eral variants of adaptive CSP [204, 247], but also adaptive

##### xDAWN [227].

##### Unsupervised adaptation of classifiers is obviously much

##### more difficult, as the class labels, hence the class-specific vari-

##### ability, is unknown. Thus, unsupervised methods have been

##### proposed to estimate the class labels of new incoming sam-

##### ples before adapting the classifier based on this estimation.

##### This technique was explored offline in [24] and [129], and

##### online in [83] for an LDA classifier and Gaussian mixture

##### model (GMM) estimation of the incoming class labels, with

##### motor imagery data. Offline, Fuzzy C-means (FCM) were

##### also explored instead of GMM to track the class means and

##### covariance for an LDA classifier [130]. Similarly, a non-linear

##### Bayesian classifier was adapted using either unsupervised

##### or semi-supervised learning (i.e. only some of the incom-

##### ing trials were labelled) using extended Kalman filtering to

##### track the changes in the class distribution parameters with

##### auto-regressive (AR) features [149]. Another simple unsuper-

##### vised adaptation of the LDA classifier for motor imagery data

##### was proposed and evaluated for both offline and online data

##### [219]. The idea was to not incrementally adapt all of the LDA

##### parameters, but only its bias, which can be estimated without

##### knowing the class labels if we know that the data is balanced,

##### i.e. with the same number of trials per class on average. This

##### approach was extended to the multiclass LDA case, and evalu-

##### ated in an offline scenario in [132].

##### Adaptation can be performed according to reinforcement

##### signals (RS), indicating whether a trial was erroneously clas-

##### sified by the BCI. Such reinforcement signals can be deduced

##### from error-related potentials (ErrP), potentials appearing fol-

##### lowing a perceived error which may have been committed by

##### either the user or the machine [68]. In [133], an incremen-

##### tal logistic regression classifier was proposed, which was

##### updated along the error gradient when a trial was judged to

##### be misclassified according to the detection of an ErrP. The

##### strength of the classifier update was also proportional to the

##### probability of this ErrP. A Gaussian probabilistic classifier

##### incorporating an RS was later proposed in [131], in which the

##### update rules of the mean and covariance of each class depend

##### on the probability of the RS. This classifier could thus incor-

##### porate a supervised, unsupervised or semi-supervised adap-

##### tation mode, according to whether the probability of the RS

##### is always correct as either 0 or 1 (supervised case), uniform,

J. Neural Eng. **15** ( 2018 ) 031005


##### i.e. uninformative (unsupervised case) or with a continuous

##### probability with some uncertainty (partially supervised case).

##### Using simulated supervised RS, this method was shown to be

##### superior to static LDA and the other supervised and unsuper-

##### vised adaptive LDA discussed above [131]. Evaluations with

##### real-world data remain to be performed. Also using ErrP in

##### offline simulations of an adaptive movement-related poten-

##### tial (MRP)-BCI, [9] augmented the training set with incom-

##### ing trials, but only with those that were classified correctly, as

##### determined by the absence of an ErrP following feedback to

##### the user. They also removed the oldest trials from the training

##### set as new trials became available. Then, the parameters of

##### the classifier, an incremental SVM, were updated based on

##### the updated training set. ErrP-based classifier adaptation was

##### explored online for code-modulated visual evoked potential

##### (c-VEP) classification in [206]. In this work, the label of the

##### incoming trial was estimated as the one decided by the clas-

##### sifier if no ErrP was detected, the opposite label otherwise

##### (for binary classification). Then, this newly labelled trial was

##### added to the training set, and the classifier and spatial filter,

##### a one-class SVM and canonical correlation analysis (CCA),

##### respectively, were retrained on the new data. Finally, [239]

##### demonstrated that classifier adaptation based on RS could also

##### be performed using classifier confidence, and that such adap-

##### tation was beneficial to P300-BCI.

##### For ERP-based BCI, semi-supervised adaptation was

##### explored with SVM and enabled the calibration of a P300-

##### speller with less data as compared to a fixed, non-adaptive clas-

##### sifier [122, 151]. This method was later tested and validated

##### online in [81]. For P300-BCI, a co-training semi-supervised

##### adaptation was performed in [178]. In this work, two classi-

##### fiers were used: a Bayesian LDA and a standard LDA. Each

##### was initially trained on training labelled data, and then used

##### to estimate the labels of unlabelled incoming data. The latter

##### were labelled with their estimated class label and used as addi-

##### tional training data to retrain the other classifier, hence the co-

##### training. This semi-supervised approach was shown offline to

##### lead to higher bit-rates than a fully supervised method, which

##### requires more supervised training data. On the other hand,

##### offline semi-supervised adaptation with an LDA as classifier

##### failed on mental imagery data, probably owing to the poor

##### robustness of the LDA to mislabelling [137]. Finally, both for

##### offline and online data, [104, 105] proposed a probabilistic

##### method to adaptively estimate the parameters of a linear clas-

##### sifier in P300-based spellers, which led to a drastic reduction

##### in calibration time, essentially removing the need for the ini-

##### tial calibration. This method exploited the specific structure of

##### the P300-speller, and notably the frequency of samples from

##### each class at each time, to estimate the probability of the most

##### likely class label. In a related work, [78] proposed a generic

##### method to adaptively estimate the parameters of the classifier

##### without knowing the true class labels by exploiting any struc-

##### ture that the application may have. Semi-supervised adapta-

##### tion was also used offline for multi-class motor imagery with

##### a Kernel discriminant analysis (KDA) classifier in [171]. This

##### method has shown its superiority over non-adaptive methods,

##### as well as over adaptive unsupervised LDA methods.

##### Vidaurre et  al , also explored co-adaptive training, where

##### both the machine and the user are continuously learning, by

##### using adaptive features and an adaptive LDA classifier [220,

##### 221]. This enabled some users who were initially unable to

##### control the BCI to achieve better than chance classification

##### performances. This work was later refined in [64] by using a

##### simpler but fully adaptive setup with auto-calibration, which

##### proved to be effective both for healthy users and for users with

##### disabilities [63]. Co-adaptive training, using adaptive CSP

##### patches, proved to be even more efficient [196].

##### Adaptive classification approaches used in BCI are sum-

##### marized in tables 1 and 2, for supervised and unsupervised

##### methods, respectively.

##### 4.1.3. Pros and cons. Adaptive classifiers were repeatedly

##### shown to be superior to non-adaptive ones for multiple types

##### of BCI, notably motor-imagery BCI, but also for some ERP-

##### based BCI. To the best of our knowledge, adaptive classifiers

##### have apparently not been explored for SSVEP-BCI. Naturally,

##### supervised adaptation is the most efficient type of adaptation,

##### as it has access to the real labels. Nonetheless unsupervised

##### adaptation has been shown to be superior to static classifiers

##### in multiple studies [24, 130, 132, 149, 219]. It can also be used

##### to shorten or even remove the need for calibration [78, 81,

##### 105, 122, 151]. There is a need for more robust unsupervised

##### adaptation methods, as the majority of actual BCI applications

##### do not provide labels, and thus can only rely on unsupervised

##### methods.

##### For unsupervised adaptation, reward signals, and notably

##### ErrP, have been exploited in multiple papers (e.g. [9, 206,

##### 239]). Note however, that ErrP decoding from EEG signals

##### may be a difficult task. Indeed, [157] demonstrated that the

##### decoding accuracy of ErrP was positively correlated with the

##### P300 decoding accuracy. This means that people who make

##### errors in the initial BCI task (here a P300), for whom error cor-

##### rection and ErrP-based adaptation would be the most useful,

```
Table 1. Summary of adaptive supervised classification methods
explored offline.
```
```
EEG pattern Features Classifier References
```
```
Motor
imagery
```
```
Band power Adaptive LDA/
QDA
```
###### [200]

```
Motor
imagery
```
```
Fractal
dimension
```
```
Adaptive LDA [96]
```
```
Motor
imagery
```
```
Band power Adaptive LDA/
QDA
```
###### [222]

```
Motor
imagery
```
```
Band power Adaptive
probabilistic NN
```
###### [86]

```
Motor
imagery
```
```
CSP Dynamic SVM
ensemble
```
###### [119]

```
Motor
imagery
```
```
Adaptive CSP SVM [204, 247]
```
```
Motor
execution
```
```
AR parameters Adaptive Gaussian
classifier
```
###### [236]

```
P300 Time points Adaptive
LDA/SVM
```
###### [227]

```
with adaptive
xDAWN
```
```
online PA classifier
```
J. Neural Eng. **15** ( 2018 ) 031005


##### have a lesser chance that the ErrP will be correctly decoded.

##### There is thus a need to identify robust reward signals.

##### Only a few of the proposed methods were actually used

##### online. For unsupervised methods, a simple and effective one

##### that demonstrated its value online in several studies is adap-

##### tive LDA, proposed by Vidaurre et  al [219]. This and other

##### methods that are based on incremental adaptation (i.e. updat-

##### ing the algorithms parameters rather than fully re-optimizing

##### them) generally have a computational complexity that is low

##### enough to be used online. Adaptive methods that require fully

##### retraining the classifier with new incoming data generally

##### have a much higher computationnal complexity (e.g. regularly

##### retraining an SVM from scratch in real-time requires a lot of

##### computing power) which might prevent them from being actu-

##### ally used online.

##### However, more online studies are clearly necessary to

##### determine how adaptation should be performed in practice,

##### with a user in the loop. This is particularly important for men-

##### tal imagery BCI in which human-learning is involved [147,

##### 170]. Indeed, because the user is adapting to the BCI by

##### learning how to perform mental imagery tasks so that they

##### are recognized by the classifier, adaptation may not always

##### help and may even be confusing to the user, as it may lead

##### to continuously-changing feedback. Both machine and human

##### learning may not necessarily converge to a suitable and stable

##### solution. A recent theoretical model of this two-learner prob-

##### lem was proposed in [168], and indicated that adaptation that

##### is either too fast or too slow can actually be detrimental to user

##### learning. There is thus a need to design adaptive classifiers

##### that ensure and favour human learning.

```
4.2. Classifying EEG matrices and tensors
```
```
4.2.1. Riemannian geometry-based classification.
```
##### Principles. The introduction of Riemannian geometry

##### in the field of BCI has challenged some of the conventions

##### adopted in the classic classification approaches; instead of

##### estimating spatial filters and/or select features, the idea of

##### a Riemannian geometry classifier (RGC) is to map the data

##### directly onto a geometrical space equipped with a suitable

##### metric. In such a space, data can be easily manipulated for

##### several purposes, such as averaging, smoothing, interpolating,

##### extrapolating and classifying. For example, in the case of EEG

##### data, mapping entails computing some form of covariance

##### matrix of the data. The principle of this mapping is based on

##### the assumption that the power and the spatial distribution of

##### EEG sources can be considered fixed for a given mental state

##### and such information can be coded by a covariance matrix.

##### Riemannian geometry studies smooth curved spaces that can

##### be locally and linearly approximated. The curved space is

##### named a manifold and its linear approximation at each point

##### is the tangent space. In a Riemannian manifold the tangent

##### space is equipped with an inner product (metric) smoothly

##### varying from point to point. This results in a non-Euclidean

##### notion of distance between any two points (e.g. each point

##### may be a trial) and a consequent notion of centre of mass of

##### any number of points (figure 2). Therefore, instead of using

##### the Euclidean distance, called the extrinsic distance, an intrin-

##### sic distance is used, which is adapted to the geometry of the

##### manifold, and thus to the manner in which the data have been

##### mapped [47, 232].

##### Amongst the most common matrix manifolds used for BCI

##### applications, we encountered the manifold of Hermitian or

##### symmetric positive definite (SPD) matrices [19] when deal-

##### ing with covariance matrices estimated from EEG trials, and

##### the Stiefel and Grassmann manifolds [62] when dealing with

##### subspaces or orthogonal matrices. Several machine learning

##### problems can be readily extended to those manifolds by tak-

##### ing advantage of their geometrical constraints (i.e. learning on

##### manifold ). Furthermore, optimization problems can be form-

##### ulated specifically on such spaces, which is leading to several

##### new optimization methods and to the solution of new prob-

##### lems [2]. Although related, manifold learning, which consists

##### of empirically attempting to locate the non-linear subspace

##### in which a dataset is defined, is different in concept and will

##### not be covered in this paper. To illustrate these notions, con-

##### sider the case of SPD matrices. The square of the intrinsic dis-

##### tance between two SPD matrices C 1 and C 2 has a closed-form

##### expression given by

**Table 2.** Summary of adaptive unsupervised classification methods
explored.

```
EEG
pattern Features Classifier References
```
```
Motor
imagery
```
```
Band power Adaptive LDA
with GMM
```
###### [24, 83, 129]

```
Motor
imagery
```
```
Band power Adaptive LDA
with FCM
```
###### [130]

```
Motor
execution
```
###### AR

```
parameters
```
```
Adaptive Gaussian
classifier
```
###### [149]

```
Motor
imagery
```
```
Band power Adaptive LDA [132, 219]
```
```
Motor
imagery
```
```
Band power Adaptive Gaussian
classifier
```
###### [131]

```
Motor
imagery
```
```
Band power Semi-supervised
CSP+LDA
```
###### [137]

```
Motor
imagery
```
```
Adaptive
band power
```
```
Adaptive LDA [63, 64, 220, 221]
```
```
Motor
imagery
```
```
Adaptive
CSP
patches
```
```
Adaptive LDA [196]
```
```
Covert
attention
```
```
Band power Incremental
logistic regression
```
###### [133]

```
MRP Band power Incremental SVM [9]
c-VEP CCA Adaptive one-class
SVM
```
###### [206]

```
P300 Time points SWLDA [239]
P300 Time points Semi-supervised
SVM
```
###### [81, 122, 151]

```
P300 Time points Co-training LDA [178]
P300 Time points Unsupervised
linear classifier
```
###### [104, 105]

```
ErrP Time points Unsupervised
linear classifier
```
###### [78]

J. Neural Eng. **15** ( 2018 ) 031005


##### δ^2 (C 1 ,C 2 )=

##### ∑

```
n
```
##### log^2 λn

##### (

##### C− 11 C 2

##### )

##### ,

##### (1)

##### where λn(M) denotes the n th eigenvalue of matrix M. For C 1

##### and C 2 SPDs, this distance is non-negative, symmetric and

##### is equal to zero if and only if C 1 =C 2. Interestingly, when

##### C 1 and C 2 are the means of two classes, the eigenvectors of

##### matrix (C− 11 C 2 ) are used to define CSP filters, while its eigen-

##### values are used for computing their Riemannian distance [47].

##### Using the distance in equation (1), the centre of mass G of a

##### set {C 1 , ...,CK} of K SPD matrices (figure 3), also called the

##### geometric mean , is the unique solution to the following optim-

##### ization problem

##### argmin

```
G
```
##### ∑

```
k
```
##### δ^2 (Ck,G).

##### (2)

##### As discussed thoroughly in [47], this definition is analogous to

##### the definition of the arithmetic mean 1 /K

##### ∑

##### kCk, which is the

##### solution of the optimization problem (2) when the Euclidean

##### distance is used instead of the Riemannian one. In contrast

##### to the arithmetic mean, the geometric mean does not have a

##### closed-form solution. A fast and robust iterative algorithm for

##### computing the geometric mean has been presented in [48].

##### The simplest RGC methods allow immediate classification of

##### trials (mapped via some form of covariance matrix) by simple

##### nearest neighbour methods, using exclusively the notion of

##### Riemannian distance (equation (1)), and possibly with the

##### notion of geometric mean (2). For instance, the Riemannian

##### minimum distance to mean (RMDM) classifier [15, 13] com-

##### putes a geometric mean for each class using training data and

##### then assigns an unlabelled trial to the class corresponding to

##### the closest mean (figure 3). Another class of RGCs consists

##### of methods projecting the data points to a tangent space fol-

##### lowed by a classification, thereafter using standard classifiers

##### such as LDA, SVM, logistic regression, etc [13, 14]. These

##### methods take advantage of both the Riemannian geometry and

##### the possibility of executing complex decision functions using

##### dedicated classifiers. An alternative approach is to project the

##### data in the tangent space, filter the data there (for example

##### by LDA), and map the data back onto the manifold to finally

##### carry out the RMDM.

##### State-of-the-art. As described above, Riemannian classifiers

##### either operate directly on the manifold (e.g. the RMDM) or by

##### the projection of the data in the tangent space. Simple RGCs

##### on the manifold have been shown to be competitive as com-

##### pared to previous state-of-the-art classifiers used in BCI as

##### long as the number of electrodes is not very large, providing

##### better robustness to noise and better generalization capabili-

##### ties, both on healthy users [13, 46, 100] and clinical popula-

##### tions [158]. RGCs based on tangent space projection clearly

##### outperformed the other state-of-the-art methods in terms of

##### accuracy [13, 14], as demonstrated by the first place they

##### have been awarded in five recent international BCI predictive

##### modelling data competitions, as reported in [47]. For a com-

##### prehensive review of the Riemannian approaches in BCI, the

##### reader can refer to [47, 232]. The various approaches using

##### Riemannian geometry classifiers for EEG-based BCIs are

##### summarized in table 3.

##### Pros and cons. As highlighted in [232], the processing pro-

##### cedures of Riemannian approaches such as RMDM is sim-

##### pler and involves fewer stages than more classic approaches.

##### Also, Riemannian classifiers apply equally well to all BCI

##### paradigms (e.g. BCIs based on mental imagery, ERPs and

##### SSVEP); only the manner in which data points are mapped in

##### the SPD manifold differs (see [47] for details). Furthermore, in

##### contrast to most classification methods, the RMDM approach

##### is parameter-free, that is, it does not require any parameter

**Figure 2.** Schematic representation of a Riemannian manifold. EEG trials are represented by points. Left: Representation of the tangent
space at point G. The shortest path on the manifold relying on two points C 1 and C 2 is named the geodesic and its length is the Riemannian
distance between them. Curves on the manifolds through a point are mapped on the tangent space as straight lines (local approximation).
Right: G represents the centre of mass (mean) of points C 1 , C 2 , C 3 and C 4. It is defined as the point minimizing the sum of the squared
distance between itself and the four points. The centre of mass is often used in RGCs as a representative for a given class.

**Figure 3.** Schematic of the Riemannian minimum distance to
mean (RMDM) classifier for a two-class problem. From training
data a centre of mass for each class is computed (G 1 and G 1 ). An
unlabelled trial (question mark) is then assigned to the class whose
centre of mass is the closest, G 1 in this example. The RMDM works
in the same manner for any dimension of the data, any number
of classes and any BCI paradigm. It does not require any spatial
filtering and feature selection, nor any parameter tuning (see text).

J. Neural Eng. **15** ( 2018 ) 031005


##### tuning, for example by cross-validation. Hence, Riemannian

##### geometry provides new tools for building simple, more robust

##### and accurate prediction models.

##### Several reasons have been proposed to advocate the use of

##### the Riemannian geometry. Due to its logarithmic nature the

##### Riemannian distance is robust to extreme values, that is, noise.

##### Also, the intrinsic Riemannian distance for SPD matrices is

##### invariant both to matrix inversion and to any linear invertible

##### transformation of the data, e.g. any mixing applied to the EEG

##### sources does not change the distances among the observed

##### covariance matrices. These properties in part explain why

##### Riemannian classification methods provide a good generaliza-

##### tion capability [224, 238], which enabled researchers to set up

##### calibration-free adaptive ERP-BCIs using simple subject-to-

##### subject and session-to-session transfer learning strategies [6].

##### Interestingly, as illustrated in [94], it is possible to not

##### only interpolate along geodesics (figure 2) on the SPD mani-

##### folds, but also to extrapolate (e.g. forecast) without leaving

##### the manifold and respecting the geometrical constraints. For

##### example, in [99] interpolation has been used for data augmen-

##### tation by generating artificial covariance matrices along geo-

##### desics but extrapolation could also have been used. Often, the

##### Riemannian interpolation is more relevant than its Euclidean

##### counterpart as it does not suffer from the so-called swelling

##### effect [232]. This effect describes the fact that a Euclidean

##### interpolation between two SPD matrices does not involve the

##### determinant of the matrix as it should (i.e. the determinant of

##### the Euclidean interpolation can exceed the determinant of the

##### interpolated matrices). In the spirit of [231], the determinant

##### of a covariance matrix can be considered as the volume of

##### the polytope described by the column of the matrix. Thus, a

##### distance that is immune to the swelling effect will respect the

##### shape of the polytope along geodesics.

##### As equation (1) indicates, computing the Riemannian dis-

##### tance between two SPD matrices involves adding squared log-

##### arithms, which may cause numerical problems; the smallest

##### eigenvalues of matrix (C

```
− 1
```
##### 1 C^2 ) tend towards zero as the num-

##### ber of electrodes increases and/or the window size for esti-

##### mating C 1 and C 2 decreases, making the logarithm operation

##### ill-conditioned and numerically unstable. Further, note that

##### the larger the dimensions, the more the distance is prone to

##### noise. Moreover, Riemannian approaches usually have high

##### computational complexities (e.g. growing cubically with the

##### number of electrodes for computing both the geometric mean

##### and the Riemannian distance). For these reasons, when the

##### number of electrodes is large with respect to the window size,

##### it is advocated to reduce the dimensions of the input matrices.

##### Classical unsupervised methods such as PCA or supervised

##### methods such as CSP can be used for this purpose. Recently,

##### Riemannian-inspired dimensionality reduction methods have

##### been investigated as well [94, 95, 189].

##### Interestingly, some approaches have tried to bridge the

##### gap between Riemannian approaches and more classical

##### paradigms by incorporating some Riemannian geometry in

##### approaches such as CSP [12, 233]. CSP was the previous

##### golden standard and is based on a different paradigm than

##### Riemannian geometry. Taking the best of those two paradigms

##### is expected to gain better robustness while compressing the

##### information.

```
4.2.2. Other matrix classifiers.
```
##### Principles. As mentioned previously, the classification

##### pipeline in BCI typically involves spatial filtering of the EEG

##### signals followed by classification of the filtered data. This

##### results in the independent optimization of several sets of

##### parameters, namely for the spatial filters and for the final clas-

##### sifier. For instance, the typical linear classifier decision func-

##### tion for an oscillatory activity BCI would be the following:

##### f(X,w,S)=

##### ∑

```
i
```
##### wilog(var(sTiX))+w 0

##### (3)

##### where X is the EEG signals matrix, w=[w 0 ,w 1 ,...,wN] is

##### the linear classifier weight vector, and S=[s 1 ,s 2 ,...,sN] is a

##### matrix of spatial filter si. Optimizing w and si separately may

##### thus lead to suboptimal solutions, as the spatial filters do not

##### consider the objective function of the classifier. Therefore, in

##### addition to RGC, several authors have shown that it is possible

##### to formulate this dual optimization problem as a single one,

##### where the parameters of the spatial filters and the linear classi-

##### fier are optimized simultaneously, with the potential to obtain

##### improved performance. The key principle of these approaches

##### is to learn classifiers (either linear vector classifiers or matrix

##### classifiers) that directly use covariance matrices as input, or

##### their vectorised version. We briefly present these approaches

##### below.

##### State-of-the-art. In [214], the EEG data were represented

##### as an augmented covariance matrix A, containing as block

##### diagonal terms both the first order term X, i.e. the signal time

##### course, and as second order terms the covariance matrices of

##### EEG trials band-pass filtered in various frequency bands. The

##### learned classifier is thus a matrix of weights W (rather than

##### a vector), with the decision function f(A,W)=〈A,W〉+b.

##### Due to the large dimensionality of the augmented covariance

##### matrix, a matrix regularization term is necessary with such

##### classifiers, e.g. to obtain sparse temporal or spatial weights.

**Table 3.** Summary of Riemannian geometry classifiers for
EEG-based BCI.

```
EEG pattern Features Classifier References
```
```
Motor
imagery
```
```
Band-pass
covariance
```
###### RMDM [13, 46]

```
Motor
imagery
```
```
Band-pass
covariance
```
```
Tangent
space + LDA
```
###### [13, 231]

```
Motor
imagery
```
```
Band-pass
covariance
```
```
SVM Riemannian
Kernel
```
###### [14]

```
P300 Special
covariance
```
###### RMDM [46]

```
P300 Special
covariance
```
###### RMDM [15]

```
P300 Special
covariance
```
###### RMDM [158]

```
SSVEP Band-pass
covariance
```
###### RMDM [34, 100]

J. Neural Eng. **15** ( 2018 ) 031005


##### Note that this approach can be applied to both ERP and oscil-

##### latory-based BCI, as the first order terms capture the temporal

##### variation, and the covariance matrices capture the EEG sig-

##### nals’ band power variations.

##### Following similar ideas in parallel, [65] represented this

##### learning problem in tensor space by constructing tensors of

##### frequency-band specific covariance matrices, which can then

##### be classified using a linear classifier as well, provided appro-

##### priate regularization is used.

##### Finally, [190] demonstrated that equation (3) can be rewrit-

##### ten as follows, if we drop the log-transform:

##### (4)f(Σ,wΣ)=vec(Σ)TwΣ+w 0

##### with wΣ=

##### ∑

##### iwivec(sis

```
T
```
##### i), Σ=X

```
T
```
##### X being the EEG cova-

##### riance matrix, and vec(M) being the vectorisation of matrix

##### M. Thus, equation (3) can be optimized directly in the space

##### of vectorised covariance matrices by optimizing the weights

##### wΣ. Here as well, owing to the usually large dimensionality

##### of vec(σ), appropriate regularization is necessary, and [190]

##### explored different approaches to do so.

##### These different approaches all demonstrated higher perfor-

##### mance than the basic CSP+LDA methods on motor imagery

##### data sets [65, 190, 214]. This suggests that such for mulations

##### can be worthy alternatives to the standard CSP+LDA

##### pipelines.

##### Pros and cons. By simultaneously optimizing spatial fil-

##### ters and classifiers, such formulations usually achieve better

##### solutions than the independent optimization of individual sets

##### of components. Their main advantage is thus increased clas-

##### sification performance. This formulation nonetheless comes

##### at the expense of a larger number of classifier weights due

##### to the high increase in dimensionality of the input features

##### (covariance matrix with (Nc∗(Nc+ 1 ))/ 2 unique values ver-

##### sus Nc values when using only the channels’ band power).

##### Appropriate regularization is thus necessary. It remains to be

##### evaluated how such methods perform for various amounts of

##### training data, as they are bound to suffer more severely from

##### the curse of dimensionality than simpler methods with fewer

##### parameters. These methods have also not been used online to

##### date. From a computational complexity point of view, such

##### methods are more demanding than traditionnal methods given

##### their increased number of parameters, as mentioned above.

##### They also generally require heavy regularization, which can

##### make their calibration longer. However, their decision func-

##### tions being linear, they should be easily applicable in online

##### scenarios. However, it remains to be seen whether they can

##### be calibrated quickly enough for online use, and what their

##### performance will be for online data.

4.2.3. Feature extraction and classification using tensors.

##### Principles. Tensors (i.e. multi-way arrays) provide a

##### natural representation for EEG data, and higher order tensor

##### decompositions and factorizations are emerging as promising

##### (but not yet very well established and not yet fully explored)

##### tools for analysis of EEG data; particularly for feature extrac-

##### tion, clustering and classification tasks in BCI [38–40, 42, 43].

##### The concept of tensorization refers to the generation of

##### higher-order structured tensors (multiway arrays) from lower-

##### order data formats, especially time series EEG data repre-

##### sented as vectors or organized as matrices. This is an essential

##### step prior to tensor (multiway) feature extraction and classifi-

##### cation [41, 42, 182].

##### The order of a tensor is the number of modes, also known

##### as ways or dimensions (e.g. for EEG BCI data: space (chan-

##### nels), time, frequency, subjects, trials, groups, conditions,

##### wavelets, dictionaries). In the simplest scenario, multichannel

##### EEG signals can be represented as a 3rd-order tensor that has

##### three physical modes: space (channel) × time × frequency.

##### In other words, S channels of EEG which are recorded over

##### T time samples, can produce S matrices of F×T dimen-

##### sional time-frequency spectrograms stacked together into an

##### F×T×S dimensional third-order tensor. For multiple trials

##### and multiple subjects, the EEG data sets can be naturally rep-

##### resented by higher-order tensors: e.g. for a 5th-order tensor:

##### space × time × frequency × trial × subject.

##### It should be noted that almost all basic vector- and matrix-

##### based machine learning algorithms for feature extraction and

##### classification have been or can be extended or generalized to

##### tensors. For example, the SVM for classification has been

##### naturally generalized to the tensor support machine (TSM),

##### Kernel TSM and higher rank TSM. Furthermore, the standard

##### LDA method has been generalized to tensor Fisher discrimi-

##### nant analysis (TFDA) and/or higher order discriminant anal-

##### ysis (HODA) [41, 183]. Moreover tensor representations of

##### BCI data are often very useful in mitigating the small sample

##### size problem in discriminative subspace selection, because the

##### information about the structure of data is often inherent in ten-

##### sors and is a natural constraint which helps reduce the number

##### of unknown feature parameters in the description of a learn-

##### ing model. In other words, when the number of EEG training

##### measurements is limited, tensor-based learning machines are

##### expected often to perform better than the corresponding vec-

##### tor- or matrix-based learning machines, as vector representa-

##### tions are associated with problems such as loss of information

##### for structured data and over-fitting for high-dimensional data.

##### State-of-the-art. To ensure that the reduced data sets contain

##### maximum information about input EEG data, we may apply

##### constrained tensor decomposition methods. For example, this

##### could be achieved on the basis of orthogonal or non-negative

##### tensor (multi-array) decompositions, or higher order (multi-

##### linear) discriminant analysis (HODA), whereby input data are

##### considered as tensors instead of more conventional vector or

##### matrix representations. In fact, tensor decomposition mod-

##### els, especially PARAFAC (also called CP decomposition),

##### TUCKER, hierarchical tucker (HT) and tensor train (TT) are

##### alternative sophisticated tools for feature extraction prob-

##### lems by capturing multi-linear and multi-aspect structures in

##### large-scale higher-order data-sets [39, 183]. Using this type of

##### approach, we first decompose multi-way data using TUCKER

##### or CP decompositions, usually by imposing specific con-

##### straints (smoothness, sparseness, non-negativity), in order

##### to retrieve basis factors and significant features from factor

##### (component) matrices. For example, wavelets/dictionaries

J. Neural Eng. **15** ( 2018 ) 031005


##### allow us to represent the data often in a more efficient way,

##### i.e. a sparse manner with different sparsity profiles [43, 183].

##### Moreover, in order to increase performance of BCI clas-

##### sification, we can apply two or more time-frequency repre-

##### sentations or the same frequency transform but with two or

##### more different parameter settings. Different frequency trans-

##### forms (or different mother wavelets) allow us to obtain differ-

##### ent sparse tensor representations with various sparsity profiles

##### and some complimentary information. For multichannel EEG

##### signals we can generate a block of at least two tensors, which

##### can be concatenated as a single data tensor: space × time ×

##### frequency × trial [43, 182, 183].

##### The key problem in tensor representation is the choice of

##### a suitable time-frequency representation (TFR) or frequency

##### transform and the selection of optimal, or close to optimal,

##### corresponding transformation parameters. By exploiting vari-

##### ous TFRs, possibly with suitably selected different parameter

##### settings for the same data, we may potentially improve the

##### classification accuracy of BCI due to additional (partially

##### redundant) information. Such approaches have been imple-

##### mented e.g. for motor imagery (MI) BCI by employing differ-

##### ent complex Morlet (Gabor) wavelets for EEG data sets with

##### 62 channels [183]. For such data sets, the authors selected dif-

##### ferent complex Morlet wavelets with two different bandwidth

##### frequency parameters fb = 1 Hz and fb = 6 Hz for the same

##### centre frequency fc = 1 Hz. For each mother wavelet the authors

##### constructed a 4th-order tensor: 62-channels × 23-frequency

##### bins × 50-time frames × 120-trials for both training and test

##### EEG data. The block of training tensor data can be concat-

##### enated as the 5th-order tensor: 62-channels × 23-frequency

##### bins × 50-time frames × 2-wavelets × 120-trials.

##### The HODA algorithm was used to estimate discriminant

##### bases. The four most significant features were selected to clas-

##### sify the data, and led to an improved accuracy higher than

##### 95%. Thus, it appears that by applying tensor decomposition

##### for suitably constructed data tensors, considerable perfor-

##### mance improvement in comparison to the standard approaches

##### can be achieved for both motor-imagery BCI [183, 223] and

##### P300 paradigms [175].

##### In this approach, transformation of data with a diction-

##### ary aims to un-correlate the raw data and express them in

##### a sparse domain. Different dictionaries (transformations)

##### contribute to obtaining different sparse representations with

##### various sparsity profiles. Moreover, augmentation of dimen-

##### sionality to create samples with additional modes improved

##### the performance.

##### To summarize, tensor decompositions with nonnegative,

##### orthonormal or discriminant bases improved the classification

##### accuracy for the BCI dataset by almost 10%. A comparison of

##### all methods mentioned is provided in table 4.

##### From the time-frequency analysis perspective, tensor

##### decompositions are very attractive, even for a single chan-

##### nel, because they simultaneously take into account temporal

##### and spectral information and variability and/or consistency of

##### time frequency representations (TFRs) for trials and/or sub-

##### jects. Furthermore, they provide links among various latent

##### (hidden) variables (e.g. temporal, spectral and spatial comp-

##### onents) often with physical or physiological meanings and

##### interpretations [40, 183].

##### Furthermore, standard canonical correlation analysis

##### (CCA) was generalized to tensor CCA and multiset CCA and

##### was successfully applied to the classification of SSVEP for

##### BCI [242, 243, 245, 246]. Tensor canonical correlation analy-

##### sis (TCCA) and its modification multiset canonical correlation

##### analysis (MsetCCA) have been one of the most efficient meth-

##### ods for frequency recognition in SSVEP-BCIs. The MsetCCA

##### method learns multiple linear transforms that implement joint

##### spatial filtering to maximize the overall correlation amongst

##### canonical variates, and hence extracts SSVEP common fea-

##### tures from multiple sets of EEG data recorded at the same stim-

##### ulus frequency. The optimized reference signals are formed by

##### combination of the common features and are completely based

##### on training data. Extensive experimental study with EEG data

##### demonstrated that the tensor and MsetCCA method improve

##### the recognition accuracy of SSVEP frequency in comparison

##### with the standard CCA method and other existing methods,

##### especially for a small number of channels and a short time

##### window length. The superior results indicate that the tensor

##### MsetCCA method is a very promising candidate for frequency

##### recognition in SSVEP-based BCIs [243].

##### Pros and cons. In summary, the recent advances in BCI

##### technologies have generated massive amounts of brain data

##### exhibiting high dimensionality, multiple modality (e.g. physi-

##### cal modes such as frequency or time, multiple brain imaging

##### techniques or conditions), and multiple couplings as func-

##### tional connectivity data. By virtue of their multi-way nature,

##### tensors provide powerful and promising tools for BCI analy-

##### sis and fusion of massive data combined with a mathemati-

##### cal backbone for the discovery of underlying hidden complex

##### (space-time-frequency) data structures [42, 183].

##### Another of their advantages is that, using tensorization and

##### low-rank tensor decomposition, they can efficiently compress

##### large multidimensional data into low-order factor matrices

##### and/or core tensors which usually represent reduced features.

##### Tensor methods can also analyze linked (coupled) blocks of

##### trials represented as large-scale matrices into the form of ten-

##### sors in order to separate common/correlated from independ-

##### ent/uncorrelated components in the observed raw EEG data.

##### Finally, it is worth mentioning that tensor decomposi-

##### tions are emerging techniques not only for feature extraction/

##### selection and BCI classification, but also for pattern recogni-

##### tion, multiway clustering, sparse representation, data fusion,

##### dimensionality reduction, coding, and multilinear blind brain

##### source separation (MBSS). They can potentially provide con-

##### venient multi-channel and multi-subject space-time-frequency

```
Table 4. Summary of tensor classifiers for EEG-based BCI.
```
```
EEG pattern Features/Methods Classifier References
```
```
Motor
imagery
```
```
Topographic map,
TFR, Connect.
```
###### LDA/HODA [183]

```
P300 Multilinear PCA SVM/TSM [223]
P300 Tine-space-freq. HODA [175]
SSVEP TCCA, MsetCCA,
Bayesian
```
###### LDA [243–246]

J. Neural Eng. **15** ( 2018 ) 031005


##### sparse representations, artefact rejection, feature extraction,

##### multi-way clustering and coherence tracking [39, 40].

##### On the cons side, the complexity of tensor methods is usu-

##### ally much higher than standard matrix and vector machine

##### learning methods. Moreover, since tensor methods are just

##### emerging as potential tools for feature extraction and clas-

##### sification, existing algorithms are not always mature and are

##### still not fully optimized. Thus, some efforts are still needed to

##### optimize and test them for real-life large scale data sets.

4.3. Transfer learning

##### 4.3.1. Principles. One of the major hypotheses in machine

##### learning is that training data, on which the classifier is trained,

##### and test data, on which the classifier is evaluated, belong to the

##### same feature space and follow the same probability distribu-

##### tion. In many applications such as computer vision, biomedi-

##### cal engineering or brain–computer interfaces, this hypothesis

##### is often violated. For BCI, a change in data distribution typi-

##### cally occurs when data are acquired from different subjects

##### and across various time sessions.

##### Transfer learning aims at coping with data that violates this

##### hypothesis by exploiting knowledge acquired while learning

##### a given task for solving a different but related task. In other

##### words, transfer learning is a set of methodologies considered

##### for enhancing performance of a learned classifier trained on

##### one task (also denoted as a domain) based on information

##### gained while learning another task. Naturally, the effective-

##### ness of transfer learning strongly depends on how well-related

##### the two tasks are. For instance, it is more relevant to perform

##### transfer learning between two P300 speller tasks performed

##### by two different subjects than between one P300 speller task

##### and a motor-imagery task performed by the same subject.

##### Transfer learning is of importance especially in situations

##### where there exists abundant labelled data for one given task,

##### denoted as a source domain, whilst data are scarce or expen-

##### sive to acquire for the second task, denoted as a target domain.

##### Indeed, in such cases, transferring knowledge from the source

##### domain to the target domain acts as a bias or as a regularizer

##### for solving the target task. We provide a more formal descrip-

##### tion of transfer learning based on the survey of Pan et al [177]

##### More formally, a domain is defined by a feature space X

##### and a marginal probability distribution P(X) where the ran-

##### dom variable X takes value X. The feature space is associated

##### with a label space Y and they are linked through a joint prob-

##### ability distribution P(X,Y) with Y=y∈Y. A task is defined

##### by a label space Y and a predictive function f(·) which

##### depends on the unknown probability distribution P(X,Y). For

##### a given task, the objective is to learn the function f(·) based

##### on pairs of examples {xi,yi}i= 1 where xi∈X and yi∈Y.

##### Define the source and target domains as respectively

##### DS={XS,PS(X)} and DT={XT,PT(X)} and the source and

##### target tasks as TS={YS,fS(·)} TT={YT,fT(·)}, respectively.

##### Hence, given the estimation of fT(·) trained based solely on

##### information from the target task, the goal of transfer learn-

##### ing is to improve on this estimation by exploiting knowledge

##### obtained from DS and TS with DS=DT or TS=TT. Note that

##### DS=DT occurs when either the feature spaces XS and XT are

##### different or when the marginal distributions PS(X) and PT(X)

##### are not equal. Similarly, TS=TT indicates that either the label

##### spaces are different or the predictive functions are different.

##### For the latter situation, this reduces to situations where the

##### two conditional probabilities differ: PS(yS|XS)=PT(yT|XT).

##### Based on the learning setting and domains and tasks, there

##### exist several situations applicable to transfer learning. For

##### instance, homogeneous transfer learning refers to cases where

##### XS=XT, and domain adaptation refers to situations where the

##### marginal probability distributions or the conditional probabil-

##### ity distributions do not match in the source and target domain.

##### Settings in which labelled data are available in both source

##### and target domains, and TS=TT, are referred to as inductive

##### transfer learning. In BCI, this may be the case when the source

##### domain and task are related to visual P300 evoked potentials

##### whilst the target domain and task involve auditory P300-

##### evoked potentials. In contrast, transductive transfer learning

##### refers to situations in which tasks are similar but domains are

##### different. A particular case is the domain adaptation problem

##### when mismatch in domains is caused by mismatch in the mar-

##### ginal or conditional probability distributions. In BCI, trans-

##### ductive transfer learning is the most frequent situation, as

##### inter-subject variability or session-to-session variability usu-

##### ally occurs. For more categorizations in transfer learning, we

##### refer the reader to the survey of Pan et al [177].

##### There exists a flurry of methods and implementations for

##### solving a transfer learning problem, which depend on spe-

##### cific situations and the application of a domain. For homoge-

##### neous transfer learning, which is the most frequent situation

##### encountered in BCIs, there exist essentially three main strat-

##### egies. If domain distributions do not match, one possible

##### strategy is to learn the transformation of source or target

##### domain data so as to correct the distribution mismatch [134,

##### 203]. If the type of mismatch occurs on the marginal distri-

##### bution, then a possible method for compensating the change

##### in distribution is to consider a reweighting scheme [208].

##### Many transfer learning approaches are also based on find-

##### ing a common feature representation for the two (or more)

##### domains. As the representation, or the retrieved latent space,

##### is common to all domains, labelled samples from the source

##### and target domain can be used to train a general classifier

##### [53, 177]. A classic strategy is to consider approaches whose

##### goal is to locate representations in which domains match.

##### Another trend for transfer learning is to consider methods

##### that learn a transformation of the data so that their distri-

##### butions match. These transformations can either be linear,

##### based for instance on kernel methods [76, 241] or non-linear,

##### through the use of an optimal transport strategy [51].

##### Note that transfer learning may not always yield enhanced

##### performance on a specific task TT. Theoretical results [55]

##### in domain adaptation and transfer learning show that gain in

##### performance on TT may be achieved only if the source and

##### target tasks are not too dissimilar. Hence, a careful analysis

##### of how well tasks relate has to be carried out before consider-

##### ing transfer learning methods. Transfer learning methods are

##### illustrated in figure 4.

J. Neural Eng. **15** ( 2018 ) 031005


##### 4.3.2. State-of-the-art. In recent years, transfer learning has

##### gained much attention for improving BCI classification. BCI

##### research has focused on transductive transfer learning, in

##### which tasks are identical between source and target. Motor

##### imagery has been the most-used paradigm to test transfer

##### learning methods, probably owing to the availability of datas-

##### ets from BCI Competitions [4, 10, 35, 67, 97, 102, 103, 143].

##### A few studies considered other paradigms such as the P300-

##### speller [74, 151, 218], and visual and spatial attention para-

##### digms [165]. A transfer learning challenge was also recently

##### organized on an error potential dataset [1].

##### Instead of considering source and target domains one-to-

##### one, a widespread strategy is to perform ensemble analyses,

##### in which many pre-recorded sessions, from possibly different

##### subjects, are jointly analysed. This addresses a well-known

##### problem in data scarcity, especially involving labelled data,

##### prone to overfitting.

##### There are many methods for combining the features and

##### classifiers within ensembles [205]. A first concern when con-

##### sidering ensembles is to guarantee the quality of the features

##### and classifiers from the source domain. Feature selection is

##### also relevant in this context (see section 2.2) to eliminate out-

##### liers. Many methods have been used to select relevant features

##### from the ensemble, for instance mutual information [186],

##### classification accuracy [143] or sparsity-inducing methods.

##### A second major challenge is to cope with the variabil-

##### ity of data across subjects or sessions. Methods from adap-

##### tive classification are sometimes applicable in the context of

##### transfer learning. Although the goal of adaptive classifica-

##### tion, as explained in section 4.1, is to update classifiers and

##### not to transfer data, transfer learning can benefit from adap-

##### tive classification to update classifiers whose initialization is

##### subject-independent. This approach has been proposed for

##### P300 classification by Lu et  al [151]. Riemannian geometry

##### can also increase robustness with respect to inter-subject and

##### inter-session variability, as demonstrated in several studies

##### [46, 238].

##### A particularly fruitful strand of research has focused on

##### building spatial filters based on ensemble data. Common spa-

##### tial patterns (CSP) and spatial filters in general are able to

##### learn quickly on appropriate training data, but do not perform

##### well with a large quantity of heterogeneous data recorded from

##### other subjects or other sessions [46]. A regularization strategy

##### in this case is effective [103]. A more relevant approach is

##### to directly regularize the CSP objective function rather than

##### the covariance matrices [143]. In this vein, Blankertz et  al

##### [21] have proposed an invariant CSP (iCSP), which regular-

##### izes the CSP objective function in a manner that diminishes

##### the influence of noise and artefacts. Fazli et  al [67] built a

##### subject-independent classifier for movement imagination

##### detection. They first extracted an ensemble of features (spatial

##### and frequency filters) and then applied LDA classifiers across

##### all subjects. They compared various ways of combining these

##### classifiers to classify a new subject’s data: simply averaging

##### their outcomes (bagging) performs adequately, but is outper-

##### formed by a sparse selection of relevant features.

##### Sparse representations are indeed relevant when applied to

##### ensemble datasets coming from multiple sessions or subjects.

##### The dictionary of waveforms / topographies / time-frequency

##### representations, from which the sparse representations are

##### derived, can be built in a manner to span a space that naturally

##### handles the session- or subject-variability. Sparsity-inducing

##### methods fall in the category of ‘invariant feature representa-

##### tion’. Dictionaries can be predefined, but to better represent

##### the data under study, they can be computed using data-driven

##### methods. Dictionary learning is a data-driven method that

##### alternatively adapts the dictionary of representative functions

##### and the coefficients of the data representation with the diction-

##### ary. Dictionary learning has been used to reveal inter-trial vari-

##### ability in neurophysiological signals [91]. Morioka et al [165]

##### proposed to learn a dictionary of spatial filters, which is then

##### adapted to the target subject. This method has the benefit of

##### taking into account the target subject’s specificities, through

##### their resting state EEG. Cho et al [35] also exploit target ses-

##### sion data by constructing spatiotemporal filters which mini-

##### mally overlap with noise patterns, an extension of Blankertz’s

##### iCSP [21].

##### An even more sophisticated method to address the domain

##### adaptation of features is to model their variability across ses-

##### sions of subjects. Bayesian models capture variability through

##### their model parameters. These models are generally imple-

##### mented in a multitask learning context, where an ensemble

##### of tasks TS={YS,fS(·)} is jointly learned from the source

##### (labelled) domain. For BCIs, ‘a task’ is typically a distinct

**Figure 4.** Illustrating the objective of domain adaptation. Left: source domain with labelled samples. Middle: target domain (with labels
and decision function for the sake of clarity). A classifier trained on the source domain will perform poorly. Right: a domain adaptation
technique will seek a common representation transformation or a mapping of domains so as to match the source and target domain
distributions.

J. Neural Eng. **15** ( 2018 ) 031005


##### recording session, either for a single or multiple subjects.

##### Bayesian models have hence been built for features in spectral

##### [4], spatial [102], and recently in combined spatial and spec-

##### tral domains [97]. Combining a Bayesian model and learning

##### from label proportion (LLP) has recently been proposed in

##### [218].

##### Another interesting domain adaptation method is to actu-

##### ally transport the features of the target data onto the source

##### domain. Once transported to the source domain, the target

##### data can then be classified with the existing classifier trained

##### on the source data. Arvaneh et  al [10] apply this approach

##### to session-to-session transfer for motor imagery BCI, by

##### estimating a linear transformation of the target data which

##### minimizes the Kullback–Leibler distance between source and

##### transformed target distributions. Recently, session-to-session

##### transfer of P300 data has been accomplished using a nonlinear

##### transform obtained by solving an optimal transport problem

##### [74]. Optimal transport is well-suited for domain adaptation

##### as its algorithms can be used for transporting probability dis-

##### tributions from one domain onto another [51]. These various

##### works are summarized in table 5.

##### 4.3.3. Pros and cons. As reported in the above cited stud-

##### ies, transfer learning is instrumental in session-to-session and

##### subject-to-subject decoding performance. This is essential to

##### be able to achieve a true calibration-free BCI mode of opera-

##### tion in the future, which in turn would improve BCI usability

##### and acceptance. In fact, it is well recognized in the community

##### that the calibration session may be unduly tiring for clinical

##### users, whose cognitive resources are limited, and annoying in

##### general for healthy users. As discussed by Sanelli et al [195],

##### receiving feedback from the very beginning of their BCI expe-

##### rience is highly motivating and engaging for novice users.

##### Transfer learning can then provide users with an adequately-

##### performing BCI, before applying co-adaptive strategies. In this

##### spirit, transfer learning may be used to initialize a BCI using

##### data from other subjects for a naive user and data from other

##### sessions for a known user. In any case such an initialization is

##### suboptimal, thus such an approach entails adapting the classi-

##### fier during the session, a topic that we have discussed in sec-

##### tion 4.1. Therefore, transfer learning and adaptivity must come

##### arm in arm to achieve the final goal of a calibration-free mode

##### of operation [46].

##### Although suboptimal in general, transfer learning is robust

##### by definition. For instance, subject-to-subject transfer learn-

##### ing can produce better results as compared to subject-specific

##### calibration if the latter is of low quality [15]. This is par-

##### ticularly useful in clinical settings, where obtaining a good

##### calibration is sometimes prohibitive [158].

##### As we have seen, the approach of seeking invariant spaces

##### for performing classification in transfer learning settings is

##### appealing theoretically and has shown promising results by

##### exploiting Riemannian geometry; however it comes at the

##### risk of throwing away some of the information that is relevant

##### for decoding. In fact, instead of coping with the variability of

##### data across sessions, as we formulated above, it may be wiser

##### to strive to benefit from the variability in the ensemble to bet-

##### ter classify the target session. The idea would be to design

##### classifiers able to represent multiple sessions or subjects.

##### The combination of transfer learning and adaptive clas-

##### sifiers represents a topic at the forefront of current research

##### in BCI. It is expected to receive increasing attention in the

##### upcoming years, leading to a much-sought new generation of

##### calibration-free BCIs.

##### Very few of the transfer learning presented methods

##### have yet been used online, but computational power is not a

##### limitation, because these methods do not require extensive

##### computational ressources, and can be run on simple desktop

##### computers. For methods whose learning phases may take a

##### long time (such as sparsity-inducing methods, or diction-

##### ary learning), this learning should be performed in advance

##### so that the adaptation to a new subject or session is time-

##### efficient [165].

```
4.4. Deep learning
```
##### Deep learning is a specific machine learning algorithm in

##### which features and the classifier are jointly learned directly

##### from data. The term deep learning is coined by the architec-

##### ture of the model, which is based on a cascade of trainable

##### feature extractor modules and nonlinearities. Owing to such

##### a cascade, learned features are usually related to increasing

##### levels of concepts. We discuss in this section  the two most

##### popular deep learning approaches for BCI: convolutional neu-

##### ral networks and restricted Boltzmann machines.

```
Table 5. Summary of transfer learning methods for BCI.
```
```
EEG
pattern
```
```
Features /
Method Classifier /Transfer References
```
```
Motor
imagery
```
```
CSP + band
power
```
```
Linear SVM [103, 143]
```
```
subject-to-subject
Motor
imagery
```
```
Sparse
feature set
```
###### LDA [67]

```
Motor
imagery
```
```
CSP Fisher LDA [35]
```
```
session-to-session
Motor
imagery
```
```
Surface
Laplacian
```
```
LDA, Bayesian
multitask
```
###### [4, 97]

```
subject-to-subject
Motor
imagery
```
```
PCSP LDA, Bayesian
model
```
###### [102]

```
multisubject
Motor
imagery
```
```
CSP + band
power
```
###### LDA [10]

```
session-to-session
Visual, Dictionary
learning
```
```
linear SVM [165]
```
```
Spatial
attention
```
```
of spatial
filters
```
```
subject-to-subject
```
```
P300 Time points Mixture of bayesian
classifiers
```
###### [218]

```
P300 Time points Fisher LDA [151]
multisubject
P300 xDAWN LDA, optimal
transport
```
###### [74]

```
session-to-session
```
J. Neural Eng. **15** ( 2018 ) 031005


4.4.1. Principles.

##### A short introduction on restricted Boltzmann machines. A

##### restricted Boltzmann machine (RBM) is a Markov random

##### field (MRF) [120] associated with a bipartite undirected

##### graph. It is composed of two sets of units: m visible ones

##### V=(V 1 ,···,Vm) and n hidden ones H=(H 1 ,···,Hn). The

##### visible units are used for representing observable data whereas

##### the hidden ones capture some dependencies between observed

##### variables. For the usual type of RBM such as those discussed

##### in this paper, units are considered as random variables that

##### take binary values (v,h) and W is a matrix whose entries wi , j

##### are the weights associated with the connection between unit vi

##### and hj. The joint probability of a given configuration (v,h) can

##### be modelled according to the probability p(v,h)=^1 Ze−E(v,h)

##### with the energy function E(v,h) being

##### E(v,h)=av−bh−vWh

##### where a and b are bias weight vectors. Note Z is a normalizing

##### factor in order that p(v,h) sums to one for all possible con-

##### figurations. Owing to the undirected bipartite graph property,

##### hidden (respective to the visible) variables are independent

##### given the visible (hidden) ones leading to:

##### p(v|h)=Πmi= 1 p(vi|h) p(h|v)=Πnj= 1 p(hj|v)

##### and marginal distributions over the visible variables can be

##### easily obtained as [69]:

##### p(v)=

##### 1

##### Z

##### ∑

```
h
```
##### e−E(v,h).

##### Hence, by optimizing all model parameters (W,b,a), it is pos-

##### sible to model the probability distribution of the observable

##### variables. Other properties of RBMS as well as connections of

##### RBMs with stochastic neural networks are detailed in [69, 90]

##### To learn the probability distribution of the input data,

##### RBMs are usually trained according to a procedure denoted as

##### contrastive divergence learning [89]. This learning procedure

##### is based on a gradient ascent of the log-likelihood of the train-

##### ing data. The derivative of the log-likelihood of an input v can

##### be easily derived [69] and the mean of this derivative over the

##### training set leads to the rule:

##### ∑

```
v
```
##### ∂L(W|v)

##### ∂wi,j

##### ∝〈vihj〉data−〈vihj〉model

##### with the two brackets respectively denoting expectation over

##### p(h|v)q(v) and over the model (p(v,h)) with q being the

##### empirical distribution of the inputs. While the first term of this

##### gradient is tractable, the second one has exponential complex-

##### ity. Contrastive divergence aims at approximating this gradi-

##### ent using a Gibbs chain procedure that computes the binary

##### state of h using p(h|v) and then obtaining an estimation of v

##### using p(v|h) [89]. There exist other methods for approximat-

##### ing the gradient of RBMs log-likelihood that may lead to bet-

##### ter solutions as well as methods for learning with continuous

##### variables [17, 213].

##### The above procedure allows one to learn a generative

##### model of the inputs using a simple layer of RBMs. A deep

##### learning strategy can be obtained by stacking several RBMs

##### with the hidden units of one layer used as inputs of the subse-

##### quent layers. Each layer is usually trained in a greedy fashion

##### [90] and fine-tuning can be performed depending on the final

##### objective of the model. A RBM is illustrated in figure 5 (left).

##### Short introduction on convolutional neural networks. A

##### convolutional neural network (ConvNet or CNN) is a feedfor-

##### ward neural network (a network in which information flows

##### uni-directionally from the input to the hidden layers to the

##### output) which has at least one convolutional layer [71, 117,

##### 117]. Such a convolutional layer maps its input to an output

##### through a convolution operator. Suppose that the input is a 1D

##### signal { xn } with N samples, its convolution through a 1D filter

##### { hm } of size M is given by:

##### y(n)=

##### M∑− 1

```
i= 0
```
##### hixn−i ∀n=0,···,N−1.

##### This equation can be extended to higher dimensions by aug-

##### menting the number of summations in accordance with the

##### dimensions. Several filters can also be independently used

##### in convolution operations leading to an increased number of

##### channels in the output. This convolutional layer is usually fol-

##### lowed by nonlinearities [75] and possibly by a pooling layer

**Figure 5.** Example architectures of two deep learning frameworks. Left: convolutional neural networks. The blue blocks refer to results of
convolving input signal with several different filters. Right: stacked restricted Boltzmann machines. Hidden layers are trained layer-wise
and the full network can be fine-tuned according to the task at hand.

J. Neural Eng. **15** ( 2018 ) 031005


##### that aggregate the local information of the output into a single

##### value, typically through an average or a max operator [25].

##### Standard ConvNet architectures usually stack several of these

##### layers (convolution + non-linearity (+ pooling)) followed

##### by other layers, typically fully connected, that act as a clas-

##### sification layer. Note however that some architectures use

##### all convolutional layers as classification layers. Given some

##### architectures, the parameters of the models are the weights of

##### all the filters used for convolution and the weights of the fully

##### connected layers.

##### ConvNets are usually trained in a supervised fashion by

##### solving an empirical risk minimization problem of the form:

##### wˆ=argmin

```
w
```
##### 1

##### 

##### ∑

```
i
```
##### L(yi,fw(xi))+Ω(w)

##### where {xi,yi}i= 1 are the training data, fw is the prediction

##### function related to the ConvNet, L(·,·) is a loss function that

##### measures any discrepancy between the true labels of xi and

##### fw(x)i, and Ω is a regularization function for the parameters

##### of the ConvNet. Owing to the specific form of the global

##### loss (average loss over the individual samples), stochastic

##### gradient descent and its variants are the most popular means

##### for optimizing deep ConvNets. Furthermore, the feedfor-

##### ward architecture of fw(·) allows the computation of the

##### gradient at any given layer using the chain rule. This can be

##### performed efficiently using the back-propagation algorithm

##### [193].

##### In several domain applications, ConvNets have been very

##### successful because they are able to learn the most relevant

##### features for the task at hand. However, their performances

##### strongly depend on their architectures and their learning

##### hyper-parameters. A ConvNet is illustrated in figure 5 (right).

##### 4.4.2. State-of-the-art. Deep neural networks (DNNs) have

##### been explored for all major types of EEG-based BCI systems;

##### that is P300, SSVEP, Motor imagery and passive BCI (for

##### emotions and workload detection). They have also been stud-

##### ied for less commonly used EEG patterns such as slow corti-

##### cal potentials (SCP) or motion-onset visual evoked potential

##### (MVEP). It is worth mentioning that all these studies were

##### performed offline.

##### Regarding P300-based BCI, Cecotti et al published the

##### very first paper which explored CNN for BCI [32]. Their

##### network comprised two convolutional layers, one to learn

##### spatial filters and the other to learn temporal filters, fol-

##### lowed by a fully connected layer. They also explored

##### ensembles of such CNNs. This network outperformed the

##### BCI competition winners on the P300-speller data set used

##### for evaluation. However, an ensemble of SVMs obtained

##### slightly better performances than the CNN approach.

##### Remaining with P300 classification, but this time in the

##### context of the rapid serial visualization paradigm (RSVP),

##### [156] explored another CNN with one spatial convolution

##### layer, two temporal convolution layers and two dense fully-

##### connected layers. They also used rectifying linear units,

##### Dropout and spatio-temporal regularization on the convo-

##### lution layers. This network was reported as more accurate

##### than a spatially Weighted LDA-PCA classifier, by 2%. It

##### was not compared to any other classifier though. It should

##### be mentioned that in this paper, as in most BCI papers on

##### deep learning, the architecture is not justified and not com-

##### pared to different architectures, apart from the fact that the

##### architecture was reported to perform well.

##### For SSVEP, [113] also explored a CNN with a spatial

##### convolutional layer and a temporal one that used band power

##### features from two EEG channels. This CNN obtained perfor-

##### mance similar to that of a three-layer MLP or that of a classi-

##### fier based on canonical correlation analysis (CCA) with kNN

##### data recorded from static users. However, it outperformed both

##### on noisy EEG data recorded from a moving user. However, the

##### classifiers that were compared to the CNN were not the state-

##### of-the-art for SSVEP classification (e.g. CCA was not used

##### with any harmonics of the SSVEP stimulus known to improve

##### performance nor with more channels).

##### For SCP classification, [59] explored a deep extreme learn-

##### ing machine (DELM), which is a multilayer ELM with the last

##### layer being a Kernel ELM. The structure of the network, its

##### number of units, the input features and hyper-parameters were

##### not justified. Such network obtained lower performance than

##### the BCI competition winners for the data set used, and was not

##### significantly better than a standard ELM or multilayer ELM.

##### For MVEP, [153] used a deep belief network (DBN)

##### composed of three RBMs. The dimensionality of the input

##### features, EEG time points, was reduced using compressed

##### sensing (CS). This DBN+CS approach outperformed a SVM

##### approach which used neither DBN nor CS.

##### Regarding passive BCIs, Yin et al explored DNNs for both

##### workload and emotions classifications [234, 235]. In [234],

##### they used adaptive DBN, composed of several stacked auto-

##### encoders (AE), for workload classification. Adaptation was

##### performed by retraining the first layer of the network using

##### incoming data labelled with their estimated class. Compared

##### to kNN, MLP or SVM, the proposed network outperformed all

##### without channel selection, but obtained similar performance

##### with feature selection. As is too often the case in DNN papers

##### for BCI, the proposed approach was not compared to the state-

##### of-the-art, e.g. to methods based on FBCSP. In [235], another

##### DBN composed of stacked AE was studied. This DNN was,

##### however, a multimodal one with separate AEs for EEG sig-

##### nals and other physiological signals. Additional layers merged

##### the two feature types. This approach appeared to outperform

##### competing classifiers and published results using the same

##### database. However, the data used to perform model selection

##### of the proposed DNN and determine its structure was all data,

##### that is, it included the test data, which biased the results.

##### Several studies have explored DNN for motor imagery

##### classification with both DBN and CNN [150, 198, 207, 210].

##### A DBN was explored in [150] to classify BP features from

##### two EEG channels. The network outperformed FBCSP and

##### the BCI competition winner but only when using an arbitrary

##### structure whose selection was not justified. When removing or

##### adding a single neuron, this network exhibited lower perfor-

##### mance than FBCSP or the competition winner, hence casting

##### doubts on its reliability and its initial structure choice. Another

##### DBN was used in [207] for motor imagery classification,

J. Neural Eng. **15** ( 2018 ) 031005


##### but was outperformed by a simple CSP+LDA classifier.

##### However, the authors proposed a method to interpret what the

##### network has learned and its decisions, which provided useful

##### insights on the possible neurophysiological causes of misclas-

##### sifications. A combination of CNN and DBN was explored in

##### [210]. They used a CNN whose output was used as input to a

##### six-layer SAE. Compared to only a CNN, a DBN or a SVM,

##### the CNN+DBN approach appeared to be the most effective. It

##### was not compared to the BCI competition winners on this data

##### set, or to other state-of-the-art methods such as Riemannian

##### geometry and FBCSP. The last study to explore DNN for

##### motor imagery is that of Schirrmeister et al [198]. This study

##### should be particularly commended as, contrary to most previ-

##### ously mentioned papers, various DNN structures are explored

##### and presented, all carefully justified and not arbitrary, and the

##### networks are rigorously compared to state-of-the-art methods.

##### They explored shallow CNN (one temporal convolution, one

##### spatial convolution, squaring and mean pooling, a softmax

##### layer), deep CNN (temporal convolution, spatial convolution,

##### then three layers of standard convolution and a softmax layer),

##### an hybrid shallow+deep CNN (i.e. their concatenation), and

##### residual NN (temporal convolution, spatial convolution, 34

##### residual layers, and softmax layer). Both the deep and shal-

##### low CNN significantly outperformed FBCSP, whereas the

##### hybrid CNN and the residual NN did not. The shallow CNN

##### was the most effective with +3.3% of classification accuracy

##### over FBCSP. The authors also proposed methods to interpret

##### what the network has learned, which can provide useful neu-

##### rophysiological insights.

##### Finally, a study explored a generic CNN, a compact one

##### with few layers and parameters, for the classification of

##### multiple EEG patterns, namely P300, movement related corti-

##### cal potentials (MRCP), ErrP and Motor imagery. This network

##### outperformed another CNN (that of [156] mentioned above),

##### and XDAWN + BDA as well as RCSP+LDA for subject-to-

##### subject classification. The parameters (number of filters and

##### the pass-band used) for xDAWN and RCSP are not specified

##### though but would be suboptimal if they used the same band

##### as for the CNN. The method is also not compared to the state-

##### of-the-art (FBCSP or Riemannian) methods. Comparison to

##### existing methods is thus again unconvincing.

##### A summary of the methods using deep learning for EEG

##### classification in BCI are listed in table 6.

##### 4.4.3. Pros and cons. DNNs have the potential to learn both

##### effective features and classifiers simultaneously from raw EEG

##### data. Given their effectiveness in other fields, DNNs certainly

##### seem promising to lead to better features and classifiers, and

##### thus to much more robust EEG classification. However, so far,

##### the vast majority of published studies on DNNs for EEG-based

##### BCIs have been rather unconvincing in demonstrating their

##### actual relevance and superiority to state-of-the-art BCI methods

##### in practice. Indeed, many studies did not compare the studied

##### DNN to state-of-the-art BCI methods or performed biased com-

##### parisons, with either suboptimal parameters for the state-of-the-

##### art competitors or with unjustified choices of parameters for

##### the DNN, which prevents us from ruling out manual tuning of

##### these parameters with knowledge of the test set. There is thus

##### a need to ensure such issues be solved in future publications

##### around DNN for BCI. An interesting exception is the work in

##### [198], who rigorously and convincingly showed that a shallow

##### CNN could outperform FBCSP. This suggests that the major

##### limitation of DNN for EEG-based BCI is that such networks

##### have a very large number of parameters, which thus requires

##### a very large number of training examples to calibrate them.

##### Unfortunately, typical BCI data sets and experiments have very

##### small numbers of training examples, as BCI users cannot be

##### asked to perform millions or even thousands of mental com-

##### mands before actually using the BCI. As a matter of fact, it has

##### been demonstrated outside the BCI field that DNNs are actu-

##### ally suboptimal and among the worst classifiers with relatively

##### small training sets [36]. Unfortunately, only small training sets

##### are typically available to design BCIs. This may explain why

##### shallow networks, which have much fewer parameters, are the

##### only ones which have proved useful for BCI. In the future, it is

##### thus necessary to either design NNs with few parameters or to

##### obtain BCI applications with very large training data bases, e.g.

##### for multi-subject classification.

##### It is also worth noting that DNNs so far were only explored

##### offline for BCI. This is owing to their very long training times.

##### Indeed, the computational complexity of DNN is generally very

##### high, both for training and testing. Calibration can take hours or

##### days on standard current computers, and testing, depending on

##### the number of layers and neurons, can also be very demanding.

##### As a result, high-performing computing tools, e.g. multiple pow-

##### erful graphic cards, may be needed to use them in practice. For

##### practical online BCI applications, the classifier has to be trained

##### in at most a few minutes to enable practical use (BCI users can-

##### not wait for half an hour or more every time they want to use the

**Table 6.** Summary of works using deep learning for EEG-based
BCI.

```
EEG pattern Features Classifier References
```
```
SCP Not
specified
```
```
Deep ELM [59]
```
```
Motion-onset
VEP
```
```
EEG time
points
```
###### DBN [153]

```
SSVEP Band power CNN [113]
P300 EEG time
points
```
###### CNN [32]

```
P300 EEG time
points
```
###### CNN [156]

```
Motor imagery Band power DBN [150]
Motor imagery/
Execution
```
```
Raw EEG CNN [198]
```
```
Motor imagery Band power DBN [207]
Motor imagery Band power CNN+DBN [210]
Workload Band power Adaptive DBN [234]
Emotions Band power DBN [235]
+zero
crossing
+entropy
ErrP, P300, EEG time
points
```
###### CNN [116]

```
MRCP, Motor
imagery
```
J. Neural Eng. **15** ( 2018 ) 031005


##### BCI). Fast training of a DNN would thus be required for BCI.

##### Designing DNNs that do not require any subject-specific train-

##### ing, i.e. a universal DNN, would be another alternative.

4.5. Other new classifiers

4.5.1. Multilabel classifiers.

##### Principles. In order to classify more than two mental

##### tasks, two main approaches can be used to obtain a multiclass

##### classification function [215]. The first approach consists in

##### directly estimating the class using multiclass techniques such

##### as decision trees, multilayer perceptrons, naive Bayes classi-

##### fiers or k-nearest neighbours. The second approach consists

##### of decomposing the problem into several binary classification

##### problems [5]. This decomposition can be accomplished in

##### different ways using (i) one-against-one pairwise classifiers

##### [20, 84], (ii) one-against-the-rest (or one-against-all) classi-

##### fiers [20, 84], (iii) hierarchical classifiers similar to a binary

##### decision tree and (iv) multi-label classifiers [154, 215]. In

##### the latter case, a distinct subset of L labels (or properties) is

##### associated to each class [58]. The predicted class is identified

##### according to the closest distance between the predicted labels

##### and each subset of labels defining a class.

##### State-of-the-art. The number of commands provided by

##### motor imagery-based BCIs depends on the number of mental

##### imagery states that the system is able to detect. This, in turn,

##### is limited by the number of body parts that users can imag-

##### ine moving in a manner that generate clear and distinct EEG

##### patterns. Multi-label approaches can thus prove useful for

##### detecting combined motor imagery tasks, i.e. imagination of

##### two or more body parts at the same time [125, 192, 226], with

##### each body part corresponding to a single label (indicating

##### whether that body part was used). Indeed, in comparison with

##### the standard approach, this approach has the advantage of

##### considerably increasing the number of different mental states

##### while using the same number of body parts: 2 P compared to

##### P , where P is the number of body parts. Thus, EEG patterns

##### during simple and combined motor imagery tasks were inves-

##### tigated to confirm the separability of seven different classes

##### of motor imagery for BCI [126, 226, 249]. For the purpose of

##### achieving continuous 3D control, both hands motor imagery

##### was adopted to complement the set of instructions in a simple

##### limb motor imagery based-BCI to go up (and rest to go down)

##### [114, 192]. The up/down control signal was the inverted addi-

##### tion of left and right autoregressive spectral amplitudes cal-

##### culated for each of the electrodes and 3 Hz-frequency bins.

##### Another method converted circular ordinal regression to a

##### multi-label classification approach to control a simulated

##### wheelchair, using data set IIIa of the third BCI competition,

##### with as motor tasks imagination of left hand, right hand,

##### foot and tongue movements [57]. Multiclass and multi-label

##### approaches have been compared to discriminate height com-

##### mands from the combination of three motor imagery tasks

##### (left hand, right hand and feet) to control a robotic arm [125].

##### A first method used a single classifier applied to the concat-

##### enated features related to each activity source (C3, Cz, C4),

##### with one source for each limb involved. A second approach

##### consisted of a hierarchical tree of three binary classifiers to

##### infer the final decision. The third approach was a combina-

##### tion of the first two approaches. All methods used the CSP

##### algorithm for feature extraction and Linear discriminant

##### analysis (LDA) for classification. All methods were validated

##### and compared to the classical one-versus-one (OVO) and

##### one-versus-rest (OVR) methods. Results obtained with the

##### hierarchical method were similar to the ones obtained with

##### the OVO and OVR approaches. The performance obtained

##### with the first approach (single classifier) and the last (com-

##### bined hierarchical classifier) were the best for all subjects.

##### The various multi-label approaches explored are mentioned

##### in table 7.

##### Pros and cons. Multiclass and multi-label approaches there-

##### fore aim to recognize more than two commands. In both

##### cases, the resulting increase in the number of recognized

##### classes potentially provides the user with a greater number

##### of commands to interact more quickly with the system, with-

##### out the need for a drop-down menu, for example. The multi-

##### label approach can make learning shorter and less tiring, as

##### it requires learning only a small number of labels. The many

##### possible combinations of these labels leads to a large number

##### of classes and therefore to more commands. In addition, the

##### multi-label approach allows redundancy in the labels describ-

##### ing a class, which can lead to better class separation. Usually,

##### the number of labels to produce is less than the number of

##### classes. Finally, as compared to standard methods, multiclass

##### and multilabel approaches usually have a lower computational

##### complexity since they can share parameters, e.g. using multi-

##### layer perceptron, or class descriptors (especially if no redun-

##### dancy is introduced).

##### However, there might be a lack of relationship between

##### the meaning of a label and the corresponding mental com-

##### mand, e.g. two hand imagery to go up. This may generate a

##### greater mental workload and therefore fatigue. It is therefore

##### necessary to choose carefully the mapping between mental

```
Table 7. Summary of multi-label (and related multiclass) approaches for EEG-based BCI.
```
```
EEG pattern Features Classifier References
```
```
Motor imagery (8 classes) Band power LDA [127]
Motor imagery (8 classes) Band power Riemannian geometry [128]
Mental tasks (4 classes) Band power cSVM [57]
Motor imagery (4 classes) Band power (Ratio between band powers) [114, 192]
Motor imagery (7 classes) Band power SVM [226]
Motor imagery (4 classes) Band power (Mapping to velocity) [162]
```
J. Neural Eng. **15** ( 2018 ) 031005


**Table 8.** Summary of classifiers that can be trained with limited
amount of data.

```
EEG pattern Features Classifier References
```
```
P300 Time points sLDA [142]
P300 Time points sLDA [22]
P300 Time points RF [3]
P300 Special covariance RMDM [46]
P300 Special covariance RMDM [15]
Motor imagery CSP + band power RF [54]
Motor imagery CSP + band power sLDA [137]
Motor imagery Band-pass covariance RMDM [14, 46]
SSVEP Band-pass covariance RMDM [100]
```
##### commands and corresponding labels. Finally, classification

##### errors remain of course possible. In particular, the set of esti-

##### mated labels may sometimes not correspond to any class, and

##### several classes may be at equal distances, thus causing class

##### confusion.

4.5.2. Classifiers that can be trained from little data.

##### Principles. As previously discussed, most EEG-based

##### BCIs are currently optimized for each subject. Indeed, this has

##### been shown to lead in general to substantially higher classifica-

##### tion performances than subject-independent classifiers. Typi-

##### cal BCI systems can be optimized by using only a few training

##### data, typically 20–100 trials per class, as subjects cannot be

##### asked to produce the same mental commands thousands of

##### times before being provided with a functional BCI. Moreover,

##### collecting such training data takes time, which is inconvenient

##### for the subjects, and an ideal BCI would thus require a cali-

##### bration time as short as possible. This calls for classifiers that

##### can be calibrated using as little training data as possible. In the

##### following we present those classifiers that were shown to be

##### effective for this purpose. They rely on using statistical estima-

##### tors dedicated to small sample size or on dividing the input

##### features between several classifiers to reduce the dimensional-

##### ity, thus reducing the amount of training data needed by each

##### classifier.

##### State-of-the-art. The three main classifiers that have been

##### shown to be effective with little training data, and thus effec-

##### tive for EEG-based BCI design, are the shrinkage LDA clas-

##### sifier [22, 137, 142], random forest [3, 54] and Riemannian

##### classifiers [47, 232].

##### The shrinkage LDA (sLDA), is a standard LDA classifier in

##### which the class-related covariance matrices used in its optim-

##### ization were regularized using shrinkage [22]. Indeed, covari-

##### ance matrices estimated from little data tend to have larger

##### extreme eigenvalues than the real data distribution, leading

##### to poor covariance estimates. This can be solved by shrink-

##### ing covariance matrices Σ as Σˆ=Σ−λI, with I the iden-

##### tity matrix, and λ the regularization parameter. Interestingly

##### enough, there are analytical solutions to automatically deter-

##### mine the best λ value (see [118]. The resulting sLDA classifier

##### has been shown to be superior to the standard LDA classifier

##### for BCI, both for ERP-based BCI [22] and for oscillatory

##### activity BCI [137]. It has also been shown that such classifier

##### can be calibrated with much fewer data than LDA to achieve

##### the same performance [137, 142]. For instance, for mental

##### imagery BCI, an sLDA has been shown to obtain similar

##### performance with ten training trials per class than a standard

##### LDA with 30 training trials per class, effectively reducing the

##### calibration time three-fold [137].

##### Random forest (RF) classifiers are ensembles of several

##### decision tree classifiers [26]. The idea behind this classifier

##### is to randomly select a subset of the available features, and to

##### train a decision tree classifier on them, then to repeat the pro-

##### cess with many random feature subsets to generate many deci-

##### sion trees, hence the name random forest. The final decision is

##### taken by combining the outputs of all decision trees. Because

##### each tree only uses a subset of the features, it is less sensitive

##### to the curse-of-dimensionality, and thus requires fewer train-

##### ing data to be effective. Outside of BCI research, among vari-

##### ous classifiers and across various classification problems and

##### domains, random forest algorithms were actually often found

##### to be among the most accurate classifiers, including problems

##### with small training data sets [26, 36]. RFs were used success-

##### fully even online both for ERP-based BCI [3] and for motor

##### imagery BCI [54]. They outperformed designs based on LDA

##### classifiers for motor imagery BCI [54].

##### Riemannian classifiers have been discussed in sec-

##### tion 4.2.1. Typically, a simple Riemannian classifier such as

##### the RMDM requires less training data as compared to optimal

##### filtering approaches such as the CSP for motor imagery [46]

##### and xDAWN for P300 [15]. This is due to the robustness of

##### the Riemannian distance, which the geometric mean inherits

##### directly, as discussed in [47]. Even more robust mean esti-

##### mations can be obtained computing Riemannian medians or

##### trimmed Riemannian means. Shrinkage and other regulariza-

##### tion strategies can also be applied to a Riemannian framework

##### to improve the estimation of covariance matrices when a small

##### number of data points is considered [100]. These methods are

##### summarized in table 8.

##### Pros and cons. sLDA, RF and the RMDM are simple classi-

##### fiers that are easy to use in practice and provide good results

##### in general, including online. We thus recommend their use.

##### sLDA and RMDM do not have any hyper-parameters, which

##### makes them very convenient to use. sLDA have been shown to

##### be superior to LDA both for ERP and oscillatory activity-based

##### BCI across a number of data sets [22, 137]. There is thus no

##### reason to use classic LDA; instead sLDA should be preferred.

##### RMDM performs as well as CSP+LDA for oscillatory activity-

##### based BCI [13, 46], as well as xDAWN+LDA but better than a

##### step-wise LDA on time samples for ERP-based BCI [15, 46] and

##### better than CCA for SSVEP [100]. Note that because LDA is a

##### linear classifier, it may be suboptimal in the hypothetic future

##### case when vast amounts of training data will be available. RF on

##### the other hand is a non-linear classifier that can be effective both

##### with small and large training sets [36]. RMDM is also non-linear

##### and performs well with small as well as large training sets [46].

##### In terms of computational complexity, while RF can be more

J. Neural Eng. **15** ( 2018 ) 031005


##### demanding than RMDM or sLDA since it uses many classifiers,

##### all of them are fairly simple and fast methods, and all have been

##### used online successfully on standard computers.

##### 5. Discussion and guidelines

##### Based on the many papers surveyed in this manuscript, we

##### identify some guidelines on whether to use various types of

##### classification methods, and if so, when and how it seems rel-

##### evant to do so. We also identify a number of open research

##### questions that deserve to be answered in order to design better

##### classification methods to make BCI more reliable and usable.

##### These guidelines and open research questions are presented in

##### the two following sections.

5.1. Summary and guidelines

##### According to the various studies surveyed in this paper, we

##### extract the following guidelines for choosing appropriate clas-

##### sification methods for BCI design:

- In terms of classification performance, adaptive classifi-

##### cation approaches, both for classifiers and spatial filters,

##### should be preferred to static ones. This should be the case

##### even if only unsupervised adaptation is possible for the

##### targeted application.

- Deep learning networks do not appear to be effective to

##### date for EEG signals classification in BCI, given the lim-

##### ited training data available. Shallow convolutional neural

##### networks are more promising.

- Shrinkage linear discriminant analysis (sLDA) should

##### always be used instead of classic LDA, as it is more effec-

##### tive and more robust for limited training data.

- When very little training data is available, transfer learning,

##### sLDA, Riemannian minimum distance to the mean

##### (RMDM) classifiers or random forest should be used.

- When tasks are similar between subjects, domain

##### adaptation can be considered for enhancing classifier per-

##### formance. However, care should be taken regarding the

##### effectiveness of the transfer learning, as it may sometimes

##### decrease performance.

- Riemannian geometry classifiers (RGC) are very prom-

##### ising, and are considered the current state-of-the-art

##### for multiple BCI problems, notably Motor imagery,

##### P300 and SSVEP classification. They should be fur-

##### ther applied and further explored to increase their

##### effectiveness.

- Tensor approaches are emerging and as such may also be

##### promising but currently require more research to be appli-

##### cable in practice, online, and to assess their performance

##### as compared to other state-of-the-art methods.

5.2. Open research questions and challenges

##### In addition to guidelines, our survey also enabled us to iden-

##### tify a number of unresolved challenges or research questions

##### and points that must be addressed. These challenges and ques-

##### tions are presented below.

- Many of the classification methods surveyed in this paper

##### have been evaluated offline only. However, an actual BCI

##### application is fundamentally online. There is thus a need

##### to study and validate these classification methods online

##### as well, to ensure they are sufficiently computationally

##### efficient to be used in real time, can be calibrated quickly

##### enough to be convenient to use and to ensure that they can

##### withstand real-life noise in EEG signals. In fact, online

##### evaluation of classifiers should be the norm rather than

##### the exception, as there is relatively little value in studying

##### classifiers for BCI if they cannot be used online.

- Transfer learning and domain adaptation may be key

##### components for calibration-free BCI. However, at this

##### stage, several efforts must be taken before they can be

##### routinely used. Among the efforts, coupling advanced

##### features such as covariance matrices and domain adap-

##### tation algorithms can further improve on the invariance

##### ability of BCI systems.

- There are also several open challenges that, once solved,

##### could make Riemannian geometry classifiers even more

##### efficient. One would be to design a stable estimator of

##### the Riemannian median to make RMDM classifiers

##### more robust to outliers than when using the Riemannian

##### mean. Another would be to work on multimodal RMDM,

##### with multiple modes per class, not just one, which could

##### potentially improve their effectiveness. Finally, there is a

##### need for methods to avoid poorly conditioned covariance

##### matrices or low rank matrices, as these could cause RGC

##### to fail.

- While deep learning approaches are lagging in per-

##### formance for BCI, mostly due to lack of large training

##### datasets, they can be strongly relevant for end-to-end

##### domain adaptation [73] or for augmenting datasets

##### through the use of generative adversarial networks [77].

- Classifiers, and the entire machine learning/signal

##### processing pipeline are not the only considerations in

##### a BCI system design. In particular, the user should be

##### considered as well and catered to so as to ensure effi-

##### cient brain–computer communications [33, 112, 144].

##### As such, future BCI classifiers should be designed to

##### ensure that users can make sense of the feedback from

##### the classifier, and can learn effective BCI control from it

##### [146].

##### 6. Conclusion

##### In this manuscript, we have surveyed the EEG classification

##### approaches that have been developed and evaluated between

##### 2007 and 2017 in order to design BCI systems. The numerous

##### approaches that were explored can be divided into four main

##### categories: adaptive classifiers, matrix and tensor classifiers,

##### transfer learning methods, and deep learning. In addition, a

##### few miscellaneous methods were identified outside these cat-

##### egories, notably the promising shrinkage LDA and random

##### forest classifiers.

##### Overall, our review revealed that adaptive classifiers, both

##### supervised and unsupervised, outperform static ones in gen-

##### eral. Matrix and tensor classifiers are also very promising

J. Neural Eng. **15** ( 2018 ) 031005


##### to improve BCI reliability; in particular, Riemannian geom-

##### etry classifiers are the current state-of-the-art for many BCI

##### designs. Transfer learning seems useful as well, particularly

##### when little training data is available, but its performance is

##### highly variable. More research should be invested to evaluate

##### it as part of standard BCI design. Shrinkage LDA and random

##### forest are also worthy tools for BCI, particularly for small

##### training data sets. Finally, contrary to their success in other

##### fields, deep learning methods have not demonstrated convinc-

##### ing and consistent improvements over state-of-the-art BCI

##### methods to date.

##### Future work related to EEG-based BCI classification

##### should focus on developing more robust and consistently

##### efficient algorithms that can be used easily and online and

##### are able to work with small training samples, noisy signals,

##### high-dimensional and non-stationary data. This could be

##### addressed by further developing transfer learning methods,

##### Riemannian geometry and tensor classifiers, and by identify-

##### ing where and how deep networks could be useful for BCI.

##### Altogether, improving those methods or defining new ones

##### should consider invariance. Indeed, an ideal classification

##### method would use features and/or classifiers that are invariant

##### over time, over users and contexts, to be effective in any situ-

##### ation. Additionally, there is also need for a new generation of

##### BCI classification methods that consider the human user in the

##### loop, i.e. that can adapt to user states, traits and skills, and pro-

##### vide feedback that the user can make sense of and learn from.

##### To conclude, this review suggests that it is time to change

##### the gold standard classification methods used in EEG-based

##### BCI so far, and apply a second generation of BCI classifiers.

##### We could for instance move from the classical CSP + LDA

##### design, mostly unchanged for years in many online studies,

##### to adaptive Riemannian geometry classifiers. Finally, and

##### even more importantly, the next generation of classification

##### approaches for EEG-based BCI will have to take the user into

##### account.

##### Acknowledgments

##### F Lotte received support from the French National

##### Research Agency with the REBEL project and grant ANR-

##### 15-CE23-0013-01, the European Research Council with the

##### BrainConquest project (grant ERC-2016-STG-714567) and

##### the Japan Society for the Promotion of Science (JSPS). A

##### Cichocki was supported by the Ministry of Education and

##### Science of the Russian Federation (grant 14.756.31.0001)

##### and the Polish National Science Center (grant 2016/20/W/

##### N24/00354). The authors would also like to acknowledge the

##### support of the BCI-LIFT Inria Project Lab.

##### ORCID iDs

##### F Lotte https://orcid.org/0000-0002-6888-9198

##### References

```
[1] BCI Challenge @ NER2015 2015 Online transfer learning
competition, at the IEEE EMBS Neural Engineering Conf.
(https://kaggle.com/c/inria-bci-challenge)
[2] Absil P A, Mahony R and Sepulchre R 2009 Optimization
Algorithms on Matrix Manifolds (Princeton, NJ: Princeton
University Press)
[3] Akram F, Han S M and Kim T S 2015 An efficient word
typing P300-BCI system using a modified T9 interface and
random forest classifier Comput. Biol. Med. 56  30 – 6
[4] Alamgir M, Grosse-Wentrup M and Altun Y 2010 Multitask
learning for brain–computer interfaces Int. Conf. on
Artificial Intelligence and Statistics pp 17– 24
[5] Allwein E L, Schapire R E and Singer Y 2001 Reducing
multiclass to binary: A unifying approach for margin
classifiers J. Mach. Learn. Res. 1  113 – 41
[6] Andreev A, Barachant A, Lotte F and Congedo M 2016
Recreational applications in OpenViBE: brain invaders and
use-the-force Brain–Computer Interfaces 2: Technology
and Application ed M Clerc et al (New York: Wiley) pp
241 – 58
[7] Ang K, Chin Z, Wang C, Guan C and Zhang H 2012 Filter
bank common spatial pattern algorithm on BCI competition
IV datasets 2a and 2b Front. Neurosci. 6
[8] Ang K K and Guan C 2013 Brain-computer interface in stroke
rehabilitation J. Comput. Sci. Eng. 7  139 – 46
[9] Artusi X, Niazi I K, Lucas M F and Farina D 2011
Performance of a simulated adaptive BCI based on
experimental classification of movement-related and error
potentials IEEE J. Emerging Sel. Top. Circuits Syst. 1  480 – 8
[10] Arvaneh M, Guan C, Ang K K and Quek C 2013 EEG data
space adaptation to reduce intersession nonstationarity in
brain–computer interface Neural Comput. 25 2146 – 71
[11] Balli T and Palaniappan R 2010 Classification of biological
signals using linear and nonlinear features Physiol. Meas.
31  903
[12] Barachant A, Bonnet S, Congedo M and Jutten C 2010
Common spatial pattern revisited by Riemannian geometry
IEEE Int. Workshop on Multimedia Signal Processing
(IEEE) pp 472– 6
[13] Barachant A, Bonnet S, Congedo M and Jutten C 2012
Multi-class brain computer interface classification
by riemannian geometry IEEE Trans. Biomed. Eng.
59  920 – 8
[14] Barachant A, Bonnet S, Congedo M and Jutten C 2013
Classification of covariance matrices using a Riemannian-
based kernel for BCI applications Neurocomputing
112  172 – 8
[15] Barachant A and Congedo M 2014 A plug&play P300 BCI
using information geometry (arXiv:1409.0107)
[16] Bashashati A, Fatourechi M, Ward R K and Birch G E 2007 A
survey of signal processing algorithms in brain–computer
interfaces based on electrical brain signals J. Neural Eng.
4  R35– 57
[17] Bengio Y, Lamblin P, Popovici D and Larochelle H 2007
Greedy layer-wise training of deep networks Advances in
Neural Information Processing Systems pp 153– 60
[18] Besserve M, Martinerie J and Garnero L 2011 Improving
quantification of functional networks with EEG inverse
problem: Evidence from a decoding point of view
NeuroImage 55 1536 – 47
[19] Bhatia R 2009 Positive Definite Matrices (Princeton, NJ:
Princeton University Press)
```
J. Neural Eng. **15** ( 2018 ) 031005


[20] Bishop M C 2006 _Pattern Recognition and Machine Learning_
(Berlin: Springer)
[21] Blankertz B, Kawanabe M, Tomioka R, Hohlefeld F,
Nikulin V and Müller K R 2008 Invariant common spatial
patterns: alleviating nonstationarities in brain–computer
interfacing _Advances in Neural Information Processing
Systems_ vol 20 (Cambridge, MA: MIT Press)
[22] Blankertz B, Lemm S, Treder M, Haufe S and Müller K R
2010 Single-trial analysis and classification of ERP
components–A tutorial _NeuroImage_ **56**  814 – 25
[23] Blankertz B, Tomioka R, Lemm S, Kawanabe M and
Müller K R 2008 Optimizing spatial filters for robust EEG
single-trial analysis _IEEE Signal Proc. Mag._ **25**  41 – 56
[24] Blumberg J, Rickert J, Waldert S, Schulze-Bonhage A,
Aertsen A and Mehring C 2007 Adaptive classification
for brain computer interfaces _29th Annual Int. Conf. of the
IEEE Engineering in Medicine and Biology Society_
pp 2536– 9
[25] Boureau Y L, Bach F, LeCun Y and Ponce J 2010
Learning mid-level features for recognition _IEEE
Conference on Computer Vision and Pattern Recognition_
pp 2559– 66
[26] Breiman L 2001 Random forests _Mach. Learn._ **45**  5 – 32
[27] Breiman L, Friedman J H, Olshen R A and Stone C J 1984
_Classification and Regression Trees_ (Monterey, CA:
Wadsworth and Brooks)
[28] Brodu N, Lotte F and Lécuyer A 2011 Comparative study
of band-power extraction techniques for motor imagery
classification _IEEE Symp. on Computational Intelligence,
Cognitive Algorithms, Mind, and Brain_ pp 1– 6
[29] Brodu N, Lotte F and Lécuyer A 2012 Exploring two novel
features for EEG-based brain–computer interfaces:
multifractal cumulants and predictive complexity
_Neurocomputing_ **79**  87 – 94
[30] Buttfield A, Ferrez P and Millan J 2006 Towards a robust BCI:
error potentials and online learning _IEEE Trans. Neural
Syst. Rehabil. Eng._ **14**  164 – 8
[31] Caramia N, Lotte F and Ramat S 2014 Optimizing spatial filter
pairs for EEG classification based on phase synchronization
_Int. Conf. on Audio, Speech and Signal Processing_
[32] Cecotti H and Graser A 2011 Convolutional neural networks
for P300 detection with application to brain–computer
interfaces _IEEE Trans. Pattern Anal. Mach. Intell._
**33**  433 – 45
[33] Chavarriaga R, Fried-Oken M, Kleih S, Lotte F and Scherer R
2017 Heading for new shores! overcoming pitfalls in BCI
design _Brain–Computer Interfaces_ **4**  60 – 73
[34] Chevallier S, Kalunga E, Elemy Q B and Yger F 2018
Riemannian classification for SSVEP-based BCI _Brain–
Computer Interfaces Handbook: Technological and
Theoretical Advances_ ed C Nam _et al_ (London: Taylor &
Francis)
[35] Cho H, Ahn M, Kim K and Chan Jun S 2015 Increasing
session-to-session transfer in a brain–computer interface
with on-site background noise acquisition _J. Neural Eng._
**12**  066009
[36] Chongsheng Z, Changchang L, Xiangliang Z and George A
2017 An up-to-date comparison of state-of-the-art
classification algorithms _Expert Syst. Appl._ **82**  128 – 50
[37] Cibas T, Soulié F F, Gallinari P and Raudys S 1994 _Variable
Selection with Optimal Cell Damage_ (London: Springer) pp
727 – 30
[38] Cichocki A 2011 Tensor decompositions: a new concept
in brain data analysis? _J. Soc. Instrum. Control Eng._
**58**  507 – 16
[39] Cichocki A, Lee N, Oseledets I, Phan A H, Zhao Q and
Mandic D 2016 Tensor networks for dimensionality
reduction and large-scale optimization: part 1 low-rank

```
tensor decompositions Found. Trends Mach. Learn.
9  249 – 429
[40] Cichocki A, Mandic D, Lathauwer L D, Zhou G, Zhao Q,
Caiafa C and Phan A H 2015 Tensor decompositions
for signal processing applications: from two-way to
multiway component analysis IEEE Signal Process. Mag.
32  145 – 63
[41] Cichocki A, Phan A H, Zhao Q, Lee N, Oseledets I,
Sugiyama M and Mandic D 2017 Tensor networks for
dimensionality reduction and large-scale optimizations. Part
2 applications and future perspectives Found. Trends Mach.
Learn. 9  431 – 673
[42] Cichocki A, Washizawa Y, Rutkowski T, Bakardjian H,
Phan A H, Choi S, Lee H, Zhao Q, Zhang L and Li Y
2008 Noninvasive BCIs: multiway signal-processing array
decompositions Computer 41
[43] Cichocki A, Zdunek R, Phan A and Amari S 2009
Nonnegative Matrix and Tensor Factorizations:
Applications to Exploratory Multi-way Data Analysis and
Blind Source Separation (New York: Wiley)
[44] Clerc M, Bougrain L and Lotte F 2016 Brain–Computer
Interfaces 1: Foundations and Methods (New York: Wiley)
[45] Clerc M, Bougrain L and Lotte F 2016 Brain–Computer
Interfaces 2: Technology and Applications (New York: Wiley)
[46] Congedo M 2013 EEG Source Analysis (Grenoble: Univ.
Grenoble Alpes)
[47] Congedo M, Barachant A and Bhatia R 2017 Riemannian
geometry for EEG-based brain–computer interfaces; a
primer and a review Brain-Comput. Interfaces 4  155 – 74
[48] Congedo M, Barachant A and Kharati K 2016 Classification
of covariance matrices using a riemannian-based kernel
for BCI applications IEEE Trans. Signal Process.
65  2211 – 20
[49] Congedo M, Lotte F and Lécuyer A 2006 Classification of
movement intention by spatially filtered electromagnetic
inverse solutions Phys. Med. Biol. 51  1971 – 89
[50] Corralejo R, Hornero R and Álvarez D 2011 Feature selection
using a genetic algorithm in a motor imagery-based
brain computer interface Annual Int. Conf. of the IEEE
Engineering in Medicine and Biology Society pp 7703– 6
[51] Courty N, Flamary R, Tuia D and Rakotomamonjy A 2017
Optimal transport for domain adaptation IEEE Trans.
Pattern Anal. Mach. Intell. 39  1853 – 65
[52] Coyle D, Principe J, Lotte F and Nijholt A 2013 Guest
editorial: brain/neuronal computer games interfaces and
interaction IEEE Trans. Comput. Intell. AI Games 5  77 – 81
[53] Daumé H III 2007 Frustratingly easy domain adaptation Proc.
of the Association Computational Linguistics
[54] David S, Reinhold S, Josef F and Müller-Putz G R 2016
Random forests in non-invasive sensorimotor rhythm
brain–computer interfaces: a practical and convenient non-
linear classifier Biomed.. Eng./Biomed. Tech. 61  77 – 86
[55] David S B, Lu T, Luu T and Pál D 2010 Impossibility
theorems for domain adaptation Proc. of the 13th Int. Conf.
on Artificial Intelligence and Statistics pp 129– 36
[56] del Millán J 2004 On the need for on-line learning in brain–
computer interfaces Proc. 2004 IEEE Int. Joint Conf. on
Neural Networks vol 4 (IEEE) pp 2877– 82
[57] Devlaminck D, Waegeman W, Bauwens B, Wyns B, Santens P
and Otte G 2010 From circular ordinal regression to
multilabel classification Preference Learning: ECML/
PKDD-10 Tutorial and Workshop p 15
[58] Dietterich T G and Bakiri G 1995 Solving multiclass learning
problems via error-correcting output codes J. Artif. Int. Res.
2  263 – 86
[59] Ding S, Zhang N, Xu X, Guo L and Zhang J 2015 Deep
extreme learning machine and its application in EEG
classification Math. Probl. Eng. 2015 129021
```
J. Neural Eng. **15** ( 2018 ) 031005


[60] Dornhege G, Blankertz B, Curio G and Müller K 2004
Boosting bit rates in non-invasive EEG single-trial
classifications by feature combination and multi-class
paradigms _IEEE Trans. Biomed. Eng._ **51**  993 – 1002
[61] Dornhege G, Blankertz B, Krauledat M, Losch F, Curio G and
Müller K R 2006 Combined optimization of spatial and
temporal filters for improving brain–computer interfacing
_IEEE Trans. Biomed. Eng._ **53**  2274 – 81
[62] Edelman A, Tomás A and Smith S T 1998 The geometry of
algorithms with orthogonality constraints _SIAM J. Matrix
Anal. Appl._ **20**  303 – 53
[63] Faller J, Scherer R, Costa U, Opisso E, Medina J and Müller-
Putz G R 2014 A co-adaptive brain–computer interface
for end users with severe motor impairment _PloS One_
**9**  e101168
[64] Faller J, Vidaurre C, Solis-Escalante T, Neuper C and
Scherer R 2012 Autocalibration and recurrent adaptation:
towards a plug and play online ERD-BCI _IEEE Trans.
Neural Syst. Rehabil. Eng._ **20**  313 – 9
[65] Farquhar J 2009 A linear feature space for simultaneous
learning of spatio-spectral filters in BCI _Neural Netw._
**22**  1278 – 85
[66] Fatourechi M, Ward R, Mason S, Huggins J, Schlogl A
and Birch G 2008 Comparison of evaluation metrics in
classification applications with imbalanced datasets _Int.
Conf. on Machine Learning and Applications_ (IEEE) pp
777 – 82
[67] Fazli S, Popescu F, Danóczy M, Blankertz B, Müller K R
and Grozea C 2009 Subject-independent mental state
classification in single trials _Neural Netw._ **22**  1305 – 12
[68] Ferrez P and Millán J 2008 Error-related EEG potentials
generated during simulated brain–computer interaction
_IEEE Trans. Biomed. Eng._ **55**  923 – 9
[69] Fischer A and Igel C 2012 An introduction to restricted
Boltzmann machines _Progress in Pattern Recognition,
Image Analysis, Computer Vision, and Applications, CIARP
2012, Lecture Notes in Computer Science_ vol 7441, ed L
Alvarez, M Mejail, L Gomez, J Jacabo (Berlin: Springer)
pp 14– 36
[70] Frey J, Appriou A, Lotte F and Hachet M 2015 Classifying
EEG signals during stereoscopic visualization to estimate
visual comfort _Computational Intelligence & Neuroscience_
**2016**  2758103
[71] Fukushima K and Miyake S 1982 Neocognitron: a self-
organizing neural network model for a mechanism of visual
pattern recognition _Competition and Cooperation in Neural
Nets_ (Berlin: Springer) pp 267– 85
[72] Gan J 2006 Self-adapting BCI based on unsupervised learning
_3rd Int. Brain–Computer Interface Workshop_
[73] Ganin Y, Ustinova E, Ajakan H, Germain P, Larochelle H,
Laviolette F, Marchand M and Lempitsky V 2016 Domain-
adversarial training of neural networks _J. Mach. Learn. Res._
**17**  1 – 35
[74] Gayraud N T, Rakotomamonjy A and Clerc M 2017 Optimal
transport applied to transfer learning for P300 detection _7th
Graz Brain–Computer Interface Conf._
[75] Glorot X, Bordes A and Bengio Y 2011 Deep sparse rectifier
neural networks _Proc. of the 14th Int. Conf. on Artificial
Intelligence and Statistics_ pp 314– 23
[76] Gong B, Shi Y, Sha F and Grauman K 2012 Geodesic flow
kernel for unsupervised domain adaptation _IEEE Conf. on
Computer Vision and Pattern Recognition_ pp 2066– 73
[77] Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-
Farley D, Ozair S, Courville A and Bengio Y 2014
Generative adversarial nets _Advances in Neural Information
Processing Systems_ pp 2672– 80
[78] Grizou J, Iturrate I, Montesano L, Oudeyer P Y and Lopes M
2014 Calibration-free BCI based control _AAAI_ pp 1213– 20

```
[79] Grosse-Wentrup M 2009 Understanding brain connectivity
patterns during motor imagery for brain–computer
interfacing Advances in Neural Information Processing
Systems vol 21
[80] Grosse-Wentrup M 2011 What are the causes of performance
variation in brain–computer interfacing? Int. J.
Bioelectromagn. 13  115 – 6
[81] Gu Z, Yu Z, Shen Z and Li Y 2013 An online semi-
supervised brain–computer interface IEEE Trans. Biomed.
Eng. 60  2614 – 23
[82] Guyon I and Elisseeff A 2003 An introduction to variable and
feature selection J. Mach. Learn. Res. 3  1157 – 82
[83] Hasan B A S and Gan J Q 2012 Hangman BCI: An
unsupervised adaptive self-paced brain–computer interface
for playing games Comput. Biol. Med. 42  598 – 606
[84] Hastie T and Tibshirani R 1997 Classification by pairwise
coupling Advances in Neural Information Processing
Systems Conf. (Denver, CO, USA, 1997) vol 10 pp 507– 13
[85] Hastie T, Tibshirani R and Friedman J 2001 The Elements of
Statistical Learning (Berlin: Springer)
[86] Hazrati M K and Erfanian A 2010 An online EEG-based
brain–computer interface for controlling hand grasp using
an adaptive probabilistic neural network Med. Eng. Phys.
32  730 – 9
[87] Herman P, Prasad G, McGinnity T and Coyle D 2008
Comparative analysis of spectral approaches to feature
extraction for EEG-based motor imagery classification
IEEE Trans. Neural Syst. Rehabil. Eng. 16  317 – 26
[88] Higashi H and Tanaka T 2013 Simultaneous design of
FIR filter banks and spatial patterns for EEG signal
classification IEEE Trans. Biomed. Eng. 60  1100 – 10
[89] Hinton G E 2002 Training products of experts by minimizing
contrastive divergence Neural Comput. 14  1771 – 800
[90] Hinton G E, Osindero S and Teh Y W 2006 A fast learning
algorithm for deep belief nets Neural Comput.
18  1527 – 54
[91] Hitziger S, Clerc M, Saillet S, Benar C and Papadopoulo T
2017 Adaptive waveform learning: a framework for
modeling variability in neurophysiological signals IEEE
Trans. Signal Process. 65  4324 – 38
[92] Hoffmann U, Vesin J and Ebrahimi T 2006 Spatial filters
for the classification of event-related potentials European
Symp. on Artificial Neural Networks
[93] Höhne J, Holz E, Staiger-Sälzer P, Müller K R, Kübler A and
Tangermann M 2014 Motor imagery for severely motor-
impaired patients: evidence for brain–computer interfacing
as superior control solution PLOS One 9  e104854
[94] Horev I, Yger F and Sugiyama M 2015 Geometry-aware
principal component analysis for symmetric positive
definite matrices M. Mach Learn 106  493 – 522
[95] Horev I, Yger F and Sugiyama M 2016 Geometry-aware
stationary subspace analysis ACML
[96] Hsu W Y 2011 EEG-based motor imagery classification
using enhanced active segment selection and adaptive
classifier Comput. Biol. Med. 41  633 – 9
[97] Jayaram V, Alamgir M, Altun Y, Scholkopf B and Grosse-
Wentrup M 2016 Transfer learning in brain-computer
interfaces IEEE Comput. Intell. Mag. 11  20 – 31
[98] Kachenoura A, Albera L, Senhadji L and Comon P 2008
ICA: a potential tool for BCI systems IEEE Signal
Process. Mag. 25  57 – 68
[99] Kalunga E, Chevallier S and Barthélemy Q 2015 Data
augmentation in Riemannian space for brain–computer
interfaces ICML Workshop on Statistics, Machine
Learning and Neuroscience (Stamlins 2015)
[100] Kalunga E, Chevallier S, Barthélemy Q, Djouani K and
Monacelli E 2016 Online SSVEP-based BCI using
riemannian geometry Neurocomputing 191  55 – 68
```
J. Neural Eng. **15** ( 2018 ) 031005


[101] Kamousi B, Liu Z and He B 2005 Classification of motor
imagery tasks for brain–computer interface applications
by means of two equivalent dipoles analysis _IEEE Trans.
Neural Syst. Rehabil. Eng._ **13**  166 – 71
[102] Kang H and Choi S 2014 Bayesian common spatial patterns
for multi-subject EEG classification _Neural Netw._
**57**  39 – 50
[103] Kang H, Nam Y and Choi S 2009 Composite common
spatial pattern for subject-to-subject transfer _IEEE Signal
Process. Lett._ **16**  683 – 6
[104] Kindermans P J, Schreuder M, Schrauwen B, Müller K R and
Tangermann M 2014 Improving zero-training brain–
computer interfaces by mixing model estimators _PLoS
One_ **9**  e102504
[105] Kindermans P J, Tangermann M, Müller K R and
Schrauwen B 2014 Integrating dynamic stopping, transfer
learning and language models in an adaptive zero-training
erp speller _J. Neural Eng._ **11**  035005
[106] Kohavi R and John G H 1997 Wrappers for feature subset
selection _Artif. Intell._ **97**  273 – 324 (Relevance)
[107] Koprinska I 2010 _Feature Selection for Brain–Computer
Interfaces_ (Berlin: Springer) pp 106– 17
[108] Krauledat M, Schröder M, Blankertz B and Müller K R 2007
Reducing calibration time for brain–computer interfaces:
a clustering approach _Advances in Neural Information
Processing Systems_ vol 19, ed B Scholkopf _et al_
(Cambridge, MA: MIT Press)
[109] Krusienski D, Grosse-Wentrup M, Galán F, Coyle D,
Miller K, Forney E and Anderson C 2011 Critical issues in
state-of-the-art brain–computer interface signal processing
_J. Neural Eng._ **8**  025002
[110] Krusienski D, McFarland D and Wolpaw J 2012 Value
of amplitude, phase, and coherence features for a
sensorimotor rhythm-based brain–computer interface
_Brain Res. Bull.._ **87**  130 – 4
[111] Krusienski D, Sellers E, Cabestaing F, Bayoudh S,
McFarland D, Vaughan T and Wolpaw J 2006 A
comparison of classification techniques for the P300
speller _J. Neural Eng._ **3**  299 – 305
[112] Kübler A, Holz E M, Riccio A, Zickler C, Kaufmann T,
Kleih S C, Staiger-Sälzer P, Desideri L, Hoogerwerf E J
and Mattia D 2014 The user-centered design as novel
perspective for evaluating the usability of BCI-controlled
applications _PLoS One_ **9**  e112392
[113] Kwak N S, Müller K R and Lee S W 2017 A convolutional
neural network for steady state visual evoked potential
classification under ambulatory environment _PloS One_
**12**  e0172578
[114] LaFleur K, Cassady K, Doud A, Shades K, Rogin E and
He B 2013 Quadcopter control in three-dimensional space
using a non-invasive motor imagery-based brain–computer
interface _J. Neural Eng._ **10**  046003
[115] Lal T, Schröder M, Hinterberger T, Weston J, Bogdan M,
Birbaumer N and Schölkopf B 2004 Support vector
channel selection in BCI _IEEE TBME_ **51**  1003 – 10
[116] Lawhern V J, Solon A J, Waytowich N R, Gordon S M,
Hung C P and Lance B J 2016 EEGNet: a compact
convolutional network for EEG-based brain–computer
interfaces (arXiv:1611.08024)
[117] LeCun Y, Boser B, Denker J S, Henderson D, Howard R E,
Hubbard W and Jackel L D 1989 Backpropagation applied
to handwritten zip code recognition _Neural Comput._
**1**  541 – 51
[118] Ledoit O and Wolf M 2004 A well-conditioned estimator
for large-dimensional covariance matrices _J. Multivariate
Anal._ **88**  365 – 411
[119] Li J and Zhang L 2010 Bilateral adaptation and
neurofeedback for brain computer interface system _J.
Neurosci. Methods_ **193**  373 – 9

```
[120] Li S Z 2009 Markov Random Field Modeling in Image
Analysis (Berlin: Springer)
[121] Li Y and Guan C 2008 Joint feature re-extraction and
classification using an iterative semi-supervised support
vector machine algorithm Mach. Learn. 71  33 – 53
[122] Li Y, Guan C, Li H and Chin Z 2008 A self-training semi-
supervised SVM algorithm and its application in an EEG-
based brain computer interface speller system Pattern
Recogn. Lett. 29  1285 – 94
[123] Liang N and Bougrain L 2012 Decoding finger flexion from
band-specific ECoG signals in humans Front. Neurosci.
6  91
[124] Lindgren J T 2017 As above, so below? Towards
understanding inverse models in BCI J. Neural Eng.
15  012001
[125] Lindig-León C 2017 Multilabel classification of EEG-
based combined motor imageries implemented for the
3D control of a robotic arm. (Classification multilabels
à partir de signaux EEG d’imaginations motrices
combinées : application au contrôle 3D d’un bras
robotique) PhD Thesis University of Lorraine, Nancy,
France
[126] Lindig-León C and Bougrain L 2015 Comparison of
sensorimotor rhythms in EEG signals during simple
and combined motor imageries over the contra and
ipsilateral hemispheres 37th Annual Int. Conf. of the IEEE
Engineering in Medicine and Biology Society (Milan, Italy)
[127] Lindig-León C and Bougrain L 2015 A multilabel
classification method for detection of combined motor
imageries IEEE Int. Conf. on Systems, Man, and
Cybernetics
[128] Lindig-León C, Gayraud N, Bougrain L and Clerc M 2016
Hierarchical classification using Riemannian geometry for
motor imagery based BCI systems BCI Meeting (Asilomar,
CA, USA, 2016)
[129] Liu G, Huang G, Meng J, Zhang D and Zhu X 2010
Improved GMM with parameter initialization for
unsupervised adaptation of brain–computer interface Int.
J. Numer. Methods Biomed. Eng. 26  681 – 91
[130] Liu G, Zhang D, Meng J, Huang G and Zhu X 2012
Unsupervised adaptation of electroencephalogram signal
processing based on fuzzy C-means algorithm Int. J.
Adapt. Control Signal Process. 26  482 – 95
[131] Llera A, Gómez V and Kappen H J 2012 Adaptive
classification on brain–computer interfaces using
reinforcement signals Neural Comput. 24  2900 – 23
[132] Llera A, Gómez V and Kappen H J 2014 Adaptive multiclass
classification for brain computer interfaces Neural
Comput. 26  1108 – 27
[133] Llera A, van Gerven M A, Gómez V, Jensen O and
Kappen H J 2011 On the use of interaction error potentials
for adaptive brain computer interfaces Neural Netw.
24  1120 – 7
[134] Long M, Wang J, Ding G, Sun J and Yu P S 2013 Transfer
feature learning with joint distribution adaptation Proc. of
the IEEE Int. Conf. on Computer Vision pp 2200– 7
[135] Lotte F 2012 A new feature and associated optimal spatial
filter for EEG signal classification: waveform length Int.
Conf. on Pattern Recognition pp 1302– 5
[136] Lotte F 2014 A tutorial on EEG signal-processing techniques
for mental-state recognition in brain–computer interfaces
Guide to Brain–Computer Music Interfacing (Berlin:
Springer) pp 133– 61
[137] Lotte F 2015 Signal processing approaches to minimize or
suppress calibration time in oscillatory activity-based
brain–computer interfaces Proc. IEEE 103  871 – 90
[138] Lotte F 2016 Towards usable electroencephalography-based
brain–computer interfaces Habilitation Thesis Habilitation
à diriger des recherches (HDR) Univ Bordeaux
```
J. Neural Eng. **15** ( 2018 ) 031005


[139] Lotte F, Bougrain L and Clerc M 2015
Electroencephalography (EEG)-based brain–computer
interfaces _Wiley Encyclopedia on Electrical and
Electronices Engineering_ (New York: Wiley)
[140] Lotte F and Congedo M 2016 _EEG Feature Extraction_ (New
York: Wiley) pp 127– 43
[141] Lotte F, Congedo M, Lécuyer A, Lamarche F and
Arnaldi B 2007 A review of classification algorithms for
EEG-based brain–computer interfaces _J. Neural Eng._
**4**  R1– 13
[142] Lotte F and Guan C 2009 An efficient P300-based brain–
computer interface with minimal calibration time _Assistive
Machine Learning for People with Disabilities Symp._
[143] Lotte F and Guan C 2011 Regularizing common spatial
patterns to improve BCI designs: Unified theory and new
algorithms _IEEE Trans. Biomed. Eng._ **58**  355 – 62
[144] Lotte F and Jeunet C 2015 Towards improved BCI based
on human learning principles _3rd Int. Brain–Computer
Interfaces Winter Conf._
[145] Lotte F and Jeunet C 2017 Online classification accuracy is
a poor metric to study mental imagery-based BCI user
learning: an experimental demonstration and new metrics
_Int. Brain–Computer Interface Conf._
[146] Lotte F, Jeunet C, Mladenovic J, N’Kaoua B and Pillette L
2018 _Signal Processing and Machine Learning for
Brain-Machine Interfaces_ ed T Toshihisa and M Arvaneh
(Stevenage, United Kingdom: Institution of Engineering
and Technology (IET))
[147] Lotte F, Larrue F and Mühl C 2013 Flaws in current human
training protocols for spontaneous brain–computer
interfaces: lessons learned from instructional design _Front.
Human Neurosci._ **7**
[148] Lotte F, Lécuyer A and Arnaldi B 2009 FuRIA: an inverse
solution based feature extraction algorithm using fuzzy set
theory for brain–computer interfaces _IEEE Trans. Signal
Process._ **57**  3253 – 63
[149] Lowne D, Roberts S J and Garnett R 2010 Sequential non-
stationary dynamic classification with sparse feedback
_Pattern Recogn._ **43**  897 – 905
[150] Lu N, Li T, Ren X and Miao H 2017 A deep learning scheme
for motor imagery classification based on restricted
Boltzmann machines _IEEE Trans. Neural Syst. Rehabil.
Eng._ **25**  566 – 76
[151] Lu S, Guan C and Zhang H 2009 Unsupervised brain
computer interface based on inter-subject information and
online adaptation _IEEE Trans. Neural Syst. Rehabil. Eng._
**17**  135 – 45
[152] Luke S 2013 _Essentials of Metaheuristics_ (Lulu) (https://
cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf)
[153] Ma T, Li H, Yang H, Lv X, Li P, Liu T, Yao D and Xu P 2017
The extraction of motion-onset VEP BCI features based
on deep learning and compressed sensing _J. Neurosci.
Methods_ **275**  80 – 92
[154] Madjarov G, Kocev D, Gjorgjevikj D and Džeroski S 2012
An extensive experimental comparison of methods for
multi-label learning _Pattern Recogn._ **45**  3084 – 104 ( _Best
Papers of Iberian Conf. on Pattern Recognition and Image
Analysis_ )
[155] Makeig S, Kothe C, Mullen T, Bigdely-Shamlo N, Zhang Z
and Kreutz-Delgado K 2012 Evolving signal processing
for brain–computer interfaces _Proc. IEEE_ **100**  1567 – 84
[156] Manor R and Geva A B 2015 Convolutional neural network
for multi-category rapid serial visual presentation BCI
_Front. Comput. Neurosci._ **9**
[157] Margaux P, Emmanuel M, Sébastien D, Olivier B and
Jérémie M 2012 Objective and subjective evaluation of
online error correction during P300-based spelling _Adv.
Human-Comput. Interact._ **2012**  4

```
[158] Mayaud L et al 2016 Brain-computer interface for the
communication of acute patients: a feasibility study and
a randomized controlled trial comparing performance
with healthy participants and a traditional assistive device
Brain-Comput. Interfaces 3  197 – 215
[159] McFarland D J, McCane L M, David S V and Wolpaw J R
1997 Spatial filter selection for EEG-based communication
Electroencephalogr. Clin. Neurophysiol. 103  386 – 94
[160] McFarland D, Sarnacki W and Wolpaw J 2011 Should the
parameters of a BCI translation algorithm be continually
adapted? J. Neurosci. Methods 199  103 – 7
[161] Meng J, Yao L, Sheng X, Zhang D and Zhu X 2015
Simultaneously optimizing spatial spectral features based
on mutual information for EEG classification IEEE Trans.
Biomed. Eng. 62  227 – 40
[162] Meng J, Zhang S, Bekyo A, Olsoe J, Baxter B and He B
2016 Noninvasive electroencephalogram based control of
a robotic arm for reach and grasp tasks Scientific Reports
6  38565
[163] Millán J, Renkens F, Mouriño J and Gerstner W 2004
Noninvasive brain-actuated control of a mobile robot by
human EEG IEEE Trans. Biomed. Eng. 51  1026 – 33
[164] Mladenovic J, Mattout J, Lotte F 2017 A generic framework
for adaptive EEG-based BCI training, operation Handbook
of Brain–Computer Interfaces ed C Nam et al (London:
Taylor & Francis)
[165] Morioka H, Kanemura A, Hirayama J I, Shikauchi M,
Ogawa T, Ikeda S, Kawanabe M and Ishii S 2015 Learning
a common dictionary for subject-transfer decoding with
resting calibration NeuroImage 111  167 – 78
[166] Mühl C, Jeunet C and Lotte F 2014 EEG-based workload
estimation across affective contexts Front. Neurosci.
8  114
[167] Mullen T, Kothe C, Chi Y M, Ojeda A, Kerth T, Makeig S,
Cauwenberghs G and Jung T P 2013 Real-time modeling and
3D visualization of source dynamics and connectivity using
wearable EEG Annual Int. Conf. of the IEEE Engineering in
Medicine and Biology Society vol 2013 pp 2184– 7
[168] Müller J S, Vidaurre C, Schreuder M, Meinecke F C, von
Bünau P and Müller K R 2017 A mathematical model for
the two-learners problem J. Neural Eng. 14  036005
[169] Müller K R, Krauledat M, Dornhege G, Curio G and
Blankertz B 2004 Machine learning techniques for
brain–computer interfaces Biomed. Technol. 49  11 – 22
[170] Neuper C and Pfurtscheller G 2010 Chapter Neurofeedback
training for BCI control Brain–Computer Interfaces:
Revolutionizing Human-Computer Interaction ed B
Graimann, G Pfurtscheller and B Allison (Berlin:
Springer) pp 65– 78
[171] Nicolas-Alonso L F, Corralejo R, Gomez-Pilar J, Álvarez D
and Hornero R 2015 Adaptive semi-supervised
classification to reduce intersession non-stationarity in
multiclass motor imagery-based brain–computer interfaces
Neurocomputing 159  186 – 96
[172] Niedermeyer E and da Silva F L 2005
Electroencephalography: Basic Principles, Clinical
Applications, and Related Fields 5th edn (Philadelphia,
PA: Lippincott Williams & Wilkins)
[173] Noirhomme Q, Kitney R and Macq B 2008 Single trial EEG
source reconstruction for brain–computer interface IEEE
Trans. Biomed. Eng. 55  1592 – 601
[174] Nurse E S, Karoly P J, Grayden D B and Freestone D R 2015
A generalizable brain–computer interface (BCI) using
machine learning for feature discovery PLoS One 10  1 – 22
[175] Onishi A, Phan A, Matsuoka K and Cichocki A 2012
Tensor classification for P300-based brain computer
interface IEEE Int. Conf. on Acoustics, Speech and Signal
Processing (IEEE) pp 581– 4
```
J. Neural Eng. **15** ( 2018 ) 031005


[176] Ortega J, Asensio-Cubero J, Gan J Q and Ortiz A 2016
Classification of motor imagery tasks for BCI with
multiresolution analysis and multiobjective feature
selection _Biomed. Eng. Online_ **15** ( Suppl. 1)
[177] Pan S J and Yang Q 2010 A survey on transfer learning _IEEE
Trans. Knowl. Data Eng._ **22**  1345 – 59
[178] Panicker R C, Puthusserypady S and Sun Y 2010
Adaptation in P300 brain–computer interfaces: a two-
classifier cotraining approach _IEEE Trans. Biomed. Eng._
**57**  2927 – 35
[179] Parzen E 1962 On estimation of a probability density
function and mode _Ann. Math. Stat._ **33**  1065 – 76
[180] Peng H, Long F and Ding C 2005 Feature selection based on
mutual information: criteria of max-dependency, max-
relevance, and min-redundancy _IEEE Trans. Pattern Anal.
Mach. Intell._ **27**  1226 – 38
[181] Pfurtscheller G, Müller-Putz G, Scherer R and Neuper C
2008 Rehabilitation with brain–computer interface
systems _IEEE Comput._ **41**  58 – 65
[182] Phan A, Cichocki A, Tichavský P, Zdunek R and Lehky S
2013 From basis components to complex structural
patterns _IEEE Int. Conf. on Acoustics, Speech and Signal
Processing_ (IEEE) pp 3228– 32
[183] Phan A H and Cichocki A 2010 Tensor decompositions for
feature extraction and classification of high dimensional
datasets _Nonlinear Theory Appl._ **1**  37 – 68
[184] Quinlan J R 1986 Induction of decision trees _Mach. Learn._
**1**  81 – 106
[185] Ramoser H, Muller-Gerking J and Pfurtscheller G 2000
Optimal spatial filtering of single trial EEG during
imagined hand movement _IEEE Trans. Rehabil. Eng._
**8**  441 – 6
[186] Ray A M _et al_ 2015 A subject-independent pattern-based
brain–computer interface _Front. Behav. Neurosci._ **9**
[187] Rivet B, Cecotti H, Phlypo R, Bertrand O, Maby E and
Mattout J 2010 EEG sensor selection by sparse spatial
filtering in P300 speller brain–computer interface _Annual
Int. Conf. of the IEEE Engineering in Medicine and
Biology Society_ (IEEE) pp 5379– 82
[188] Rivet B, Souloumiac A, Attina V and Gibert G 2009 xDAWN
algorithm to enhance evoked potentials: application
to brain computer interface _IEEE Trans. Biomed. Eng._
**56**  2035 – 43
[189] Rodrigues P, Bouchard F, Congedo M and Jutten C 2017
Dimensionality reduction for BCI classification using
Riemannian geometry _7th Graz Brain–Computer Interface
Conf. (Graz, Austria_ , _September 2017)_
[190] Roijendijk L, Gielen S and Farquhar J 2016 Classifying
regularized sensor covariance matrices: an alternative
to CSP _IEEE Trans. Neural Syst. Rehabil. Eng._
**24**  893 – 900
[191] Roy R N, Charbonnier S, Campagne A and Bonnet S
2016 Efficient mental workload estimation using task-
independent EEG features _J. Neural Eng._ **13**  026019
[192] Royer A S, Doud A J, Rose M L and He B 2010 EEG control
of a virtual helicopter in 3-dimensional space using
intelligent control strategies _IEEE Trans. Neural Syst.
Rehabil. Eng._ **18**  581 – 9
[193] Rumelhart D E _et al_ 1988 Learning representations by back-
propagating errors _Cogn. Model._ **5**  1
[194] Samek W, Kawanabe M and Müller K R 2014 Divergence-
based framework for common spatial patterns algorithms
_IEEE Rev. Biomed. Eng._ **7**  50 – 72
[195] Sannelli C, Vidaurre C, Müller K R and Blankertz B 2011
CSP patches: an ensemble of optimized spatial filters. an
evaluation study _J. Neural Eng._ **8** 025012
[196] Sannelli C, Vidaurre C, Müller K R and Blankertz B
2016 Ensembles of adaptive spatial filters increase

```
BCI performance: an online evaluation J. Neural Eng.
13  046003
[197] Schettini F, Aloise F, Aricò P, Salinari S, Mattia D and
Cincotti F 2014 Self-calibration algorithm in an
asynchronous P300-based brain–computer interface J.
Neural Eng. 11  035004
[198] Schirrmeister R T, Springenberg J T, Fiederer L D J,
Glasstetter M, Eggensperger K, Tangermann M,
Hutter F, Burgard W and Ball T 2017 Deep learning with
convolutional neural networks for EEG decoding and
visualization Human Brain Mapp.
[199] Schlögl A, Kronegg J, Huggins J and Mason S G 2007
Chapter: Evaluation criteria in BCI research Towards
Brain–Computer Interfacing (Cambridge, MA: MIT
Press) pp 327– 42
[200] Schlögl A, Vidaurre C and Müller K R 2010 Adaptive
methods in BCI research-an introductory tutorial Brain–
Computer Interfaces (Berlin: Springer) pp 331– 55
[201] Seno B D, Matteucci M and Mainardi L 2008 A genetic
algorithm for automatic feature extraction in P300
detection IEEE Int. Joint Conf. on Neural Networks pp
3145 – 52
[202] Shenoy P, Krauledat M, Blankertz B, Rao R and Müller K R
2006 Towards adaptive classification for BCI J. Neural
Eng. 3  R13
[203] Si S, Tao D and Geng B 2010 Bregman divergence-based
regularization for transfer subspace learning IEEE Trans.
Knowl. Data Eng. 22  929 – 42
[204] Song X, Yoon S C and Perera V 2013 Adaptive common
spatial pattern for single-trial EEG classification in
multisubject BCI Int. IEEE/EMBS Conf. on Neural
Engineering pp 411– 4
[205] Soria-Frisch A 2012 A critical review on the usage of
ensembles for BCI Towards Practical Brain–Computer
Interfaces ( Biological, Medical Physics and Biomedical
Engineering ) ed B Z Allison et al (Berlin: Springer) pp
41 – 65
[206] Spüler M, Rosenstiel W and Bogdan M 2012 Online
adaptation of a c-VEP brain–computer interface (BCI)
based on error-related potentials and unsupervised
learning PloS One 7  e51077
[207] Sturm I, Lapuschkin S, Samek W and Müller K R 2016
Interpretable deep neural networks for single-trial EEG
classification J. Neurosci. Methods 274  141 – 5
[208] Sugiyama M, Nakajima S, Kashima H, Buenau P V and
Kawanabe M 2008 Direct importance estimation with
model selection and its application to covariate shift
adaptation Advances in Neural Information Processing
Systems pp 1433– 40
[209] Sykacek P, Roberts S J and Stokes M 2004 Adaptive BCI
based on variational bayesian kalman filtering: an
empirical evaluation IEEE Trans. Biomed. Eng. 51  719 – 29
[210] Tabar Y R and Halici U 2016 A novel deep learning approach
for classification of EEG motor imagery signals J. Neural
Eng. 14  016003
[211] Thiyam D B, Cruces S, Olias J and Cichocki A 2017
Optimization of alpha-beta log-det divergences and their
application in the spatial filtering of two class motor
imagery movements Entropy 19  89
[212] Thomas E, Dyson M and Clerc M 2013 An analysis of
performance evaluation for motor-imagery based bci J.
Neural Eng. 10  031001
[213] Tieleman T 2008 Training restricted boltzmann machines
using approximations to the likelihood gradient Proc..
of the 25th Int. Conf. on Machine Learning (ACM) pp
1064 – 71
[214] Tomioka R and Müller K R 2010 A regularized
discriminative framework for EEG analysis with
```
J. Neural Eng. **15** ( 2018 ) 031005


application to brain–computer interface _Neuroimage_
**49**  415 – 32
[215] Tsoumakas G and Katakis I 2007 Multilabel classification:
an overview _Int. J. Data Warehousing Mining_ **3**  1 – 13
[216] van Erp J, Lotte F and Tangermann M 2012 Brain-computer
interfaces: beyond medical applications _IEEE Comput._
**45**  26 – 34
[217] Vaughan T, McFarland D, Schalk G, Sarnacki W,
Krusienski D, Sellers E and Wolpaw J 2006 The
wadsworth BCI research and development program: at
home with BCI _IEEE Trans. Neural Syst. Rehabil. Eng._
**14**  229 – 33
[218] Verhoeven T, Hübner D, Tangermann M, Müller K R,
Dambre J and Kindermans P J 2017 True zero-training
brain–computer interfacing—an online study _J. Neural
Eng._ **14**  036021
[219] Vidaurre C, Kawanabe M, Von Bunau P, Blankertz B and
Muller K 2011 Toward unsupervised adaptation of LDA
for brain–computer interfaces _IEEE Trans. Biomed. Eng._
**58**  587 – 97
[220] Vidaurre C, Sannelli C, Müller K R and Blankertz B 2011
Co-adaptive calibration to improve BCI efficiency
_J. Neural Eng._ **8**  025009
[221] Vidaurre C, Sannelli C, Müller K R and Blankertz B 2011
Machine-learning-based coadaptive calibration for
brain–computer interfaces _Neural Comput._ **23**  791 – 816
[222] Vidaurre C, Schlögl A, Cabeza R, Scherer R and
Pfurtscheller G 2007 Study of on-line adaptive
discriminant analysis for EEG-based brain computer
interfaces _IEEE Trans. Biomed. Eng._ **54**  550 – 6
[223] Washizawa Y, Higashi H, Rutkowski T, Tanaka T and
Cichocki A 2010 Tensor based simultaneous feature extraction
and sample weighting for EEG classification _Int. Conf. on
Neural Information Processing, ICONIP 2010: Neural
Information Processing. Models and Applications_ (Berlin:
Springer) pp 26– 33
[224] Waytowich N, Lawhern V, Bohannon A, Ball K and Lance B
2016 Spectral transfer learning using information
geometry for a user-independent brain–computer interface
_Front. Neurosci._ **10**  430
[225] Wei Q, Wang Y, Gao X and Gao S 2007 Amplitude and phase
coupling measures for feature extraction in an EEG-based
brain–computer interface _J. Neural Eng._ **4**  120
[226] Yi W, Qiu S, Qi H, Zhang L, Wan B and Ming D 2013
EEG feature comparison and classification of simple and
compound limb motor imagery _J. Neuroeng. Rehabil._
**10** 106
[227] Woehrle H, Krell M M, Straube S, Kim S K, Kirchner E A
and Kirchner F 2015 An adaptive spatial filter for
user-independent single trial detection of event-related
potentials _IEEE Trans. Biomed. Eng._ **62**  1696 – 705
[228] Wolpaw J, Birbaumer N, McFarland D, Pfurtscheller G
and Vaughan T 2002 Brain-computer interfaces for
communication and control _Clin. Neurophysiol._
**113**  767 – 91
[229] Wolpaw J and Wolpaw E 2012 _Brain–Computer Interfaces:
Principles and Practice_ (Oxford: Oxford University Press)
[230] Wolpaw J R, McFarland D J, Neat G W and Forneris C A
1991 An EEG-based brain–computer interface for
cursor control _Electroencephalogr. Clin. Neurophysiol._
**78**  252 – 9
[231] Yger F 2013 A review of kernels on covariance matrices
for BCI applications _IEEE Int. Workshop on Machine
Learning for Signal Processing_ pp 1– 6
[232] Yger F, Berar M and Lotte F 2017 Riemannian approaches in
brain–computer interfaces: a review _IEEE Trans. Neural
Syst. Rehabil. Eng._ **25**  1753 – 62

```
[233] Yger F, Lotte F and Sugiyama M 2015 Averaging covariance
matrices for EEG signal classification based on the CSP:
an empirical study 23rd European Signal Processing Conf.
pp 2721– 5
[234] Yin Z and Zhang J 2017 Cross-session classification of
mental workload levels using EEG and an adaptive deep
learning model Biomed. Signal Process. Control 33  30 – 47
[235] Yin Z, Zhao M, Wang Y, Yang J and Zhang J 2017
Recognition of emotions using multimodal physiological
signals and an ensemble deep learning model Comput.
Methods Programs Biomed. 140  93 – 110
[236] Yoon J W, Roberts S J, Dyson M and Gan J Q 2009
Adaptive classification for brain computer interface
systems using sequential monte carlo sampling Neural
Netw. 22  1286 – 94
[237] Zander T and Kothe C 2011 Towards passive brain–computer
interfaces: applying brain–computer interface technology
to human-machine systems in general J. Neural Eng. 8
025005
[238] Zanini P, Congedo M, Jutten C, Said S and Berthoumieu Y
2017 Transfer learning: a Riemannian geometry
framework with applications to brain–computer interfaces
IEEE Trans. Biomed. Eng.
[239] Zeyl T, Yin E, Keightley M and Chau T 2016 Partially
supervised P300 speller adaptation for eventual stimulus
timing optimization: target confidence is superior to error-
related potential score as an uncertain label J. Neural Eng.
13  026008
[240] Zhang H, Chavarriaga R and Millán J D R 2015
Discriminant brain connectivity patterns of performance
monitoring at average and single-trial levels NeuroImage
120  64 – 74
[241] Zhang K, Zheng V, Wang Q, Kwok J, Yang Q and Marsic I
2013 Covariate shift in hilbert space: a solution via sorrogate
kernels Int. Conference on Machine Learning pp 388– 95
[242] Zhang Y, Zhou G, Jin J, Wang M, Wang X and Cichocki A
2013 L1-regularized multiway canonical correlation
analysis for SSVEP-based BCI IEEE Trans. Neural Syst.
Rehabil. Eng. 21  887 – 96
[243] Zhang Y, Zhou G, Jin J, Wang X and Cichocki A 2014
Frequency recognition in SSVEP-based BCI using
multiset canonical correlation analysis Int. J. Neural Syst.
24  1450013
[244] Zhang Y, Zhou G, Jin J, Wang X and Cichocki A 2015
Optimizing spatial patterns with sparse filter bands
for motor-imagery based brain–computer interface
J. Neurosci. Methods 255  85 – 91
[245] Zhang Y, Zhou G, Jin J, Zhang Y, Wang X and Cichocki A
2017 Sparse Bayesian multiway canonical correlation
analysis for EEG pattern recognition Neurocomputing
225  103 – 10
[246] Zhang Y, Zhou G, Zhao Q, Onishi A, Jin J, Wang X
and Cichocki A 2011 Multiway canonical correlation
analysis for frequency components recognition in
SSVEP-based BCIs Neural Information Processing
(Berlin: Springer)
[247] Zhao Q, Zhang L, Cichocki A and Li J 2008 Incremental
common spatial pattern algorithm for BCI IEEE Int. Joint
Conf. on Neural Networks
[248] Zhou S M, Gan J Q and Sepulveda F 2008 Classifying
mental tasks based on features of higher-order statistics
from EEG signals in brain–computer interface Inf. Sci.
178  1629 – 40
[249] Zhou Z, Wan B, Ming D and Qi H 2010 A novel technique
for phase synchrony measurement from the complex
motor imaginary potential of combined body and limb
action J. Neural Eng. 7 046008
```
J. Neural Eng. **15** ( 2018 ) 031005


