# Recerca post-TFG: millorar les prediccions dins la finestra d'estímul

Branca `post-tfg-research`. Abast: només la finestra d'estímul (0–1,5 s), sense
pre-estímul. Protocol idèntic al del TFG:

- **Partició:** per a cada subjecte, el 80 % inicial dels trials (cronològic) és
  desenvolupament i el 20 % final és el holdout.
- **Selecció:** les configuracions s'escullen només amb dades de desenvolupament.
- **Holdout:** es puntua una sola vegada.

Codi:

- `src/br_analysis/research/`: benchmark local.
- `gpu/`: models profunds a Kaggle.

## 1. Resultats fins ara (holdout, balanced accuracy mitjana)

### C vs IP (conflicte interocular)

| Model | Holdout | Guany vs Riemann (aparellat, mateixos 46 subjectes) | Subjectes p<0,05 |
|---|---:|---:|---:|
| Riemann per subjecte (TFG) | 0,550 | — | 11 |
| xDAWN + TS, ERP-LDA, subfinestres, ensembles | 0,542–0,551 | ≈ 0 | 4–9 |
| **EEGNet conjunt + EA + fine-tuning** | **0,622** | **+0,071 (p = 0,0003)** | 20 |
| **EEG-Inception conjunt** | **0,621** (AUC 0,689) | **+0,069 (p = 0,0001)** | 20 |
| EEGNeX conjunt | 0,616 | +0,065 (p = 0,0009) | 14 |
| ShallowNet conjunt | 0,580 | +0,028 (p = 0,047) | 12 |
| CBraMod preentrenat (fine-tuning) | 0,543 | −0,008 | 6 |
| *Ensemble post-hoc dels 4 primers (orientatiu)* | *0,638* | *+0,087* | — |

Comprovacions fetes:

- Els holdouts són idèntics trial a trial als del benchmark local.
- El model conjunt s'entrena només amb el 70 % inicial de cada subjecte.
- Falta un subjecte (P069 a C vs IP) que no té les dues classes en totes les particions.

**Control de color.** Si es fixa el color reportat (CR vs IPR, CG vs IPG), Riemann dona
0,548 i 0,546, el mateix nivell que C vs IP. Per tant, l'efecte no depèn del color que la
persona diu que ha vist.

### IM vs IP (com es resol el conflicte)

Tots els models queden entre 0,49 i 0,52 i cap no supera Riemann de manera
significativa, ni els profunds. Diagnòstics:

- **IM és la classe majoritària.** La mediana és del 69 % dels trials incongruents, amb un
  rang del 19 % al 92 % segons el subjecte.
- **El detector de conflicte no hi veu res.** Un model C vs IP aplicat a trials IM vs IP
  dona AUC 0,498, és a dir, els trials "mixed" no semblen ni més ni menys conflictius.
- **Únic indici temporal:** 0,25–0,50 s (dev CV 0,517, p = 0,026 sense corregir per les
  6 finestres). És el període en què la rivalitat es resol a l'inici de l'estímul.
- **Els subjectes amb classes més equilibrades tendeixen a donar millors resultats**
  (Spearman 0,30, p = 0,047).

## 2. Interpretació

1. **Entrenar amb tots els subjectes alhora és el que marca la diferència.** Per subjecte
   només hi ha uns 60 trials de holdout i uns 250 d'entrenament, i la variació entre
   subjectes és sobretot soroll de mostreig. Un model conjunt amb Euclidean Alignment fa
   servir uns 9 000 trials. Les variants clàssiques per subjecte (filtres, xDAWN,
   finestres) no milloren res.
2. **Els models fundacionals no ajuden aquí.** CBraMod queda per sota dels models petits
   supervisats, en línia amb els benchmarks recents d'ERP i dades petites
   (EEG-FM-Compass, OmniEEG-Bench).
3. **IM vs IP és un problema diferent.** L'estímul és físicament idèntic i només canvia
   el resultat perceptiu. La resposta és subjectiva i depèn sobretot de l'estat previ a
   l'estímul, que hem deixat fora de l'abast. Si hi ha senyal dins l'estímul, ha de ser
   feble i primerenc.

## 3. Cribratge a GPU (successive halving, ~40 configuracions per tasca)

`gpu/kaggle_screen.ipynb` canvia una sola cosa cada vegada respecte d'EEGNet + EA +
augmentació:

- 16 arquitectures;
- filtres i resolució;
- 5 finestres dins l'estímul;
- només canals posteriors, i sense EA;
- augmentació, mixup, label smoothing i optimització;
- multitasca amb la tercera classe.

Les rondes eliminatòries fan servir només validació, que durant el cribratge és el 25 %
final del desenvolupament. Els 3 finalistes de cada tasca, i el seu ensemble, passen el
protocol complet. Ha trigat 1,6 h (C vs IP) i 1,1 h (IM vs IP), en paral·lel. Cap
configuració ha fallat.

### C vs IP: una millora més, la multitasca

| Model (holdout, 46 subjectes) | Bal. acc. | AUC | vs Riemann | vs EEGNet conjunt |
|---|---:|---:|---:|---:|
| Riemann per subjecte (TFG) | 0,551 | 0,620 | — | — |
| EEGNet conjunt (1a tanda) | 0,622 | 0,669 | +0,071 (p = 0,0003) | — |
| **EEGNet + 3a classe IM (`aux3`)** | **0,641** | **0,697** | **+0,089 (p < 0,0001)** | +0,019 (p = 0,12) |
| CTNet + 3a classe IM | 0,632 | 0,689 | +0,081 (p < 0,0001) | +0,010 |
| **Ensemble de finalistes (escollits a validació)** | **0,646** | **0,699** | **+0,095 (p < 0,0001)** | +0,024 (p = 0,055) |

- **Multitasca.** L'única millora que supera la base de manera consistent és afegir els
  trials IM com a tercera classe d'entrenament (més dades i més estructura). És coherent
  en validació (+0,012 d'AUC amb 3 llavors) i en holdout (+1,9 punts).
- **Negatius robustos, que perden ≥0,03 d'AUC de validació:**
  - escurçar la finestra, perquè l'efecte d'alfa és tardà i cal tota la finestra de 0–1,5 s;
  - treure el passa-alt d'1 Hz;
  - un learning rate més baix;
  - baixar a 64 Hz;
  - ATCNet, Deep4, SCCNet, Shallow, EEGSimpleConv i Conformer.
- **Neutres** (dins el soroll entre llavors, SD ≈ 0,014): canals posteriors, 250 Hz,
  weight decay, mixup, mida del batch i sense EA. En aquest darrer cas l'AUC és igual però
  la balanced accuracy baixa.
- **Prevalença:** l'estadístic mínim d'Allefeld continua donant γ0 = 0, perquè amb uns 60
  trials de holdout per subjecte és massa conservador. La prova de grup (mitjana > 0,5, i
  millora aparellada) és clarament significativa.

### IM vs IP: resultat negatiu robust

- **Ronda 1:** totes les configuracions queden entre 0,46 i 0,54 d'AUC de validació.
- **Ronda 2:** les millors tornen cap a 0,50–0,52 amb més llavors, l'efecte típic de
  regressió a la mitjana.
- **Holdout dels finalistes:** canals posteriors 0,497, passa-baix 15 Hz 0,503, EEG-TCNet
  0,510 i ensemble 0,509. Cap no supera Riemann (0,499) ni el que s'esperaria per atzar.
- **La multitasca no ajuda:** afegir C com a tercera classe empitjora el resultat (−0,016
  d'AUC).

Amb 16 arquitectures, 5 finestres i totes les variants de pipeline, **dins la finestra
d'estímul no hi ha senyal descodificable de com es resol el conflicte**, almenys amb
aquestes dades i aquesta mida mostral. És coherent amb la literatura: el resultat
perceptiu de la rivalitat depèn sobretot de l'estat previ a l'estímul, que queda fora de
l'abast.

## 4. Recomanacions

1. **C vs IP:**
   - El model de referència nou és EEGNet + EA + multitasca amb 3 classes: **64 %**, contra
     el 55 % del TFG.
   - Marge restant: una tanda final amb un ensemble divers (EEGNet-aux3, CTNet-aux3,
     EEG-Inception, EEGNeX), 5 llavors i reentrenament amb tot el desenvolupament (no
     només el 70 %). Previsiblement afegiria +0,5–1,5 punts.
   - A partir d'aquí el límit és el soroll de mostreig d'uns 60 trials de holdout per
     subjecte, no el model.
2. **IM vs IP:** no val la pena gastar més GPU dins la finestra d'estímul. Si interessa
   continuar, cal canviar la pregunta:
   - incloure el pre-estímul;
   - o analitzar les diferències individuals en la taxa de respostes "mixed", que va del
     19 % al 92 % segons el subjecte.

## 5. IM vs IP: exploració barata i àmplia (CPU, sense holdout)

Codi a `src/br_analysis/research/im_explore.py`. Les característiques s'extreuen una
sola vegada per trial dels FIF originals (−1,24 a 2,0 s, 500 Hz). Cada prova és una
regressió logística entrenada amb tots els subjectes alhora, amb z-score per subjecte
calculat només amb dades de desenvolupament.

### Troballa 1: el comportament prediu més que l'EEG

Els codes 3/5/7 i 4/6/8 corresponen a les dues configuracions ull–color. Per subjecte
estan equilibrades (ràtio 0,80–1,36).

Una regressió logística amb dues fonts d'informació:
- la configuració de l'estímul;
- la història de respostes (si el trial anterior, i l'incongruent anterior, van ser
  "mixed").

Resultats al holdout:

| Línia base de comportament | Holdout |
|---|---:|
| Configuració + història | **0,563** (p = 2·10⁻⁵) |
| Només història | 0,532 (p = 3·10⁻⁴) |
| Només configuració | 0,525 (n.s.) |

Tots els models d'EEG provats abans (0,49–0,52) queden per sota.

### Troballa 2: l'EEG gairebé no hi aporta res

Es van provar 27 famílies de característiques: potència per bandes de delta a gamma alta,
per finestra, de l'alfa posterior i de la gamma alta frontal; ERP en franges de 50 ms; i
covariàncies a l'espai tangent. També es van mirar el pre-estímul i el post-estímul.

- **Totes queden entre 0,47 i 0,54 d'AUC de validació.**
- **Cap aporta un guany significatiu per sobre del comportament**: el màxim és +0,015, amb
  p ≥ 0,19.
- **L'únic indici coherent és a 1,0–1,5 s.** Amb validació creuada per blocs sobre tot el
  desenvolupament es replica:

  | Característica | AUC |
  |---|---:|
  | Espai tangent, 1,0–1,5 s | 0,521 (p = 0,010) |
  | Potència per bandes, 1,0–1,5 s | 0,520 (p = 0,004) |
  | ERP post-estímul, 1,5–2,0 s | 0,523 (p = 0,013) |
  | *Control: espai tangent, 0–0,5 s* | *0,485* |

- **La corba d'aprenentatge és plana.** Amb un 25 % dels subjectes s'obté 0,515, amb un
  50 % 0,522 i amb el 100 % 0,521. Més dades o un model més gran no ajudarien.
- **La finestra no val la pena ampliar-la.** El pre-estímul (−0,5–0 s) no aporta res amb
  aquestes característiques (0,47–0,50).

### Conclusió

Al final de l'estímul hi ha un senyal real però molt petit (AUC ≈ 0,52, que equival a
≈ 51–52 % de balanced accuracy). Encaixa amb la preparació de la decisió o la resposta més
que amb la percepció mateixa, però amb aquestes dades no es pot distingir. No té sentit
gastar GPU en aquesta tasca:

- la corba d'aprenentatge és plana;
- els models profunds ja havien fallat;
- el comportament (configuració + història) és un predictor més fort que qualsevol
  característica d'EEG.

Si es vol continuar, el millor camí seria canviar la pregunta: modelar la història
perceptiva o les diferències individuals en la taxa de respostes "mixed", del 19 % al 92 %
segons el subjecte.
