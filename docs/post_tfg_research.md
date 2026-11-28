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

## 6. Riemann clàssic entrenat amb tots els subjectes (CPU)

Per separar l'efecte d'entrenar conjuntament de l'efecte del model profund: la mateixa
pipeline Riemann del TFG, però entrenada amb tots els subjectes alhora i Euclidean
Alignment, al mateix holdout.

| C vs IP (holdout) | Bal. acc. | AUC | Comparació aparellada |
|---|---:|---:|---|
| Riemann per subjecte (TFG) | 0,550 | 0,620 | — |
| **Riemann conjunt + EA** | **0,599** | **0,644** | vs per subjecte: +0,049 bal. acc. (p = 0,013), +0,025 AUC (n.s., p = 0,59) |
| EEGNet conjunt | 0,622 | 0,669 | Riemann conjunt −0,024 (p = 0,14) |
| Ensemble de finalistes GPU | 0,646 | 0,699 | Riemann conjunt −0,048 (p = 0,005), AUC −0,056 (p = 0,002) |

12 de 47 subjectes significatius (p < 0,05).

- **Entrenar conjuntament també ajuda el model clàssic**, però sobretot en la
  balanced accuracy. L'AUC gairebé no canvia. Per tant, el guany ve més de calibrar
  millor el llindar amb moltes dades que d'ordenar millor els trials.
- **Els models profunds conjunts aporten un guany real per sobre**: amb multitasca i
  ensemble, +5 punts de balanced accuracy i +0,056 d'AUC respecte del Riemann conjunt,
  tots dos significatius.
- Aproximadament la meitat de la millora total (55 % → 64,6 %) s'explica per l'entrenament
  conjunt, i l'altra meitat pel model profund.

**IM vs IP.** El Riemann conjunt dona 0,507 de balanced accuracy (AUC 0,513). No supera
l'atzar (p = 0,89) ni el Riemann per subjecte (+0,006, p = 0,76), i només 2 de 47
subjectes són significatius. L'entrenament conjunt no canvia el resultat negatiu.

## 7. IM vs IP: neurociència i comportament (sense descodificar)

Codi a `src/br_analysis/research/im_neuro.py`; figures a `docs/images/post_tfg/` (generades amb
`python -m br_analysis.research.post_figures`, en versió clara i fosca). Són anàlisis de grup,
interpretables, que pregunten *on* i *quan* hi ha diferències, i què explica qui respon "mixed".

### Mapes espectrals (figura 2)

Per a cada subjecte, la d de Cohen per finestra × banda × sensor. Prova de grup amb una sola
permutació de clústers sobre tota la graella (finestres × bandes × sensors), és a dir, una sola
prova per contrast.

- **C vs IP (control positiu):** un sol clúster gran (p = 0,001). Sota conflicte hi ha menys potència
  alfa i beta (8–30 Hz) als sensors posteriors, a partir de 0,5 s, i el clúster ja té una mica de beta
  a 0–0,5 s. Replica l'efecte del TFG i de l'estudi original.
- **IM vs IP:** cap clúster (el millor té p = 0,61) en cap banda ni finestra, tampoc al pre-estímul.

### ERP (figura 5)

IM i IP se superposen gairebé del tot a les regions occipital, parietal i frontal. L'única diferència
és un clúster parietal a 0,98–1,08 s (p = 0,023), sense corregir per les 3 regions (corregit,
p ≈ 0,07). És coherent amb el senyal feble de 1,0–1,5 s de la secció 5.

### Model d'equilibri interocular (figura 3)

Les dues configuracions (A i B) intercanvien quin ull veu cada color. Model ordinal d'una sola
variable per subjecte: un equilibri latent (avantatge de l'ull + avantatge del color) més soroll,
tallat a ±θ. Entre −θ i θ la resposta és "mixed". El model prediu més "mixed" a la configuració més
equilibrada.

- **Prova no circular:** el desequilibri s'estima només amb les respostes pures (vermell contra verd),
  que són trials diferents dels "mixed". Com més desequilibrada és una configuració, menys "mixed" hi
  ha: Spearman ρ = −0,37, p = 0,010, amb el 66 % dels subjectes en la direcció predita.
- **Limitacions:** el model no s'ajusta del tot a 9 dels 47 subjectes (en conjunt, p < 10⁻⁵), així que
  és una bona primera aproximació però no ho explica tot. θ (l'amplada de la franja "mixed") explica
  gairebé tota la diferència de taxa entre persones (ρ = 0,92), però això és gairebé una tautologia.

### Comportament (figura 4)

- **Inèrcia perceptiva:** després d'un trial "mixed", els dos trials incongruents següents tenen
  +6 punts més de probabilitat de ser "mixed", per sobre d'un barrejat per blocs que conserva la deriva
  lenta (p < 10⁻⁶). A partir del tercer trial desapareix. (Als desfasaments de 9–12 trials surt un
  petit efecte negatiu que probablement és un artefacte del barrejat per blocs de 20.)
- **Deriva dins la sessió:** la taxa de "mixed" puja del 60 % al 73 % durant el primer terç de la
  sessió i després s'estabilitza al voltant del 66–70 %.
- **Configuració:** 64,8 % de "mixed" a A i 69,2 % a B a nivell de grup (p = 0,053). A nivell
  individual la diferència mitjana és de 9 punts.

### Diferències individuals (figura 4, dreta)

La taxa de "mixed" (del 19 % al 92 %) **no** es relaciona amb:

- la dominància ocular;
- la freqüència alfa individual, ni amb l'alçada o la potència del pic alfa (espectre del pre-estímul,
  sensors posteriors);
- la descodificabilitat del conflicte (AUC de C vs IP per subjecte);
- la inèrcia perceptiva.

Totes les correlacions tenen |ρ| < 0,16, amb p de Holm = 1.

### Conclusió

L'EEG de cuir cabellut porta una empremta clara de *si* els ulls estan en conflicte (alfa i beta
posteriors), però gairebé res de *com* es resol. El que prediu la resolució és conductual: quin ull
veu quin color (el balanç interocular) i què ha respost la persona fa un moment.

Sobre theta: cap sensor significatiu en C vs IP. El pic més alt durant l'estímul és t ≈ 4,0 a FT7
(frontotemporal, propens a soroll muscular), i al pre-estímul n'hi ha un de t ≈ 3,6 (CP1), quan
encara no hi ha conflicte: és nivell de soroll. El theta frontal clàssic de conflicte és de
resposta i control (Stroop, flanker); aquí la resposta és diferida i el conflicte és perceptiu, de
nivell baix. A més, una finestra de 0,5 s només conté 2–3 cicles de theta.

## 8. Calibratge: quantes dades de la persona nova calen? (C vs IP, CPU)

`python -m br_analysis.research.calibration c_vs_ip`. Versió ràpida del Riemann conjunt:
covariàncies, EA en espai de covariància (R C R), espai tangent a la identitat i LDA amb shrinkage.
Per a cada persona: entrenament **només amb els altres 46**, cadascun alineat amb tota la seva
sessió. De la persona nova s'agafen els primers k trials, només per alinear (sense etiquetes) o
també afegits amb etiquetes. Puntuació sempre sobre el 20 % final.

| Dades de la persona nova | Només alinear | També amb etiquetes |
|---|---|---|
| cap (sense alinear) | 54,7 % | – |
| 1 trial | 54,6 % | – |
| 5 trials | 57,9 % | 58,7 % |
| 20 trials | 58,5 % | 58,5 % |
| 10 % de la sessió | 58,4 % | 58,9 % |
| 20 % | **59,9 %** | 60,3 % |
| 50 % | 60,8 % | 60,7 % |
| 80 % | 60,8 % | 59,8 % |

- Amb 5 trials sense etiquetar ja es guanyen ~3 punts; amb el 20 % de la sessió, ~5, i la corba
  s'aplana. Tots els guanys respecte de no alinear són significatius (Wilcoxon, p < 0,006).
- Les etiquetes de la persona nova no hi afegeixen res: tot el guany és l'alineament.
- L'AUC és gairebé constant (≈ 0,635) des d'1 trial: l'alineament sobretot recol·loca el llindar
  de decisió de cada persona, més que millorar la separació entre classes.
- Correcció d'una xifra anterior: el 53,4 % per a subjectes no vistos venia d'un classificador
  MDM log-euclidià. Amb espai tangent + LDA, el mateix escenari dona **60,8 %**, al nivell del model
  conjunt que sí que veu les etiquetes de la persona.

### L'altre eix: quants altres subjectes calen?

`python -m br_analysis.research.calibration c_vs_ip --sources` (amb `--shard i/3` per repartir-ho en
3 processos). Per a cada persona, N altres subjectes a l'atzar (repetit 5 cops per N ≤ 5 i 3 per
N = 10, 20), més les dades de la persona nova: el 80 % sense etiquetes (només alinear), o el primer
20 % / 80 % amb etiquetes. N = 0 és el model per subjecte (només dades pròpies).

| Altres subjectes | 80 % sense etiquetes | 20 % etiquetat | 80 % etiquetat |
|---|---|---|---|
| 0 | – | 51,0 % (n.s.) | 54,0 % |
| 1 | 52,0 % | 51,9 % | 54,5 % |
| 2 | 53,8 % | 53,2 % | 54,9 % |
| 5 | 55,1 % | 55,1 % | 56,1 % |
| 10 | 54,4 % | 55,5 % | 56,3 % |
| 20 | 57,3 % | 57,2 % | 58,4 % |
| 46 | **60,8 %** | 60,2 % | 59,8 % |

- La corba **no s'aplana** a 46: més participants probablement ajudarien més que més trials per
  persona.
- Les etiquetes pròpies només compten quan hi ha pocs subjectes (amb 0–2, el 80 % etiquetat guanya
  1–3 punts); amb 46, les tres variants convergeixen.
- El model només amb dades pròpies (54,0 %) és coherent amb el Riemann per subjecte (55,0 %).

Figura: `docs/images/post_tfg/fig6_calibration.png`.

## 9. Altres comprovacions

### El clúster de theta fronto-medial de la tesi no es manté amb els 47 subjectes

`python -m br_analysis.research.theta_cluster`. Test de clústers canal × temps (5000 permutacions),
C vs IP, finestra 0–1,5 s, tots els subjectes.

- La tesi reportava un clúster de theta a Fz i FCz (1,17–1,24 s), trobat amb un subconjunt de
  subjectes i amb la cerca restringida a Fz, FCz i Cz. La xifra de la tesi no és coherent: la taula
  diu p corregida = 0,024, però les figures i el text diuen 0,1.
- Amb els 47 subjectes, restringit a Fz, FCz i Cz, **no es forma cap clúster** (ni amb llindar 0,01 ni
  0,05). La direcció és la mateixa (IP > C), però el t més alt és 1,8 a Fz (~1,0 s, p sense corregir
  = 0,08): una tendència.
- A tot el cuir cabellut, el theta tampoc arriba (millor clúster posterior, p = 0,16).
- L'alfa sí que es manté: clúster ampli posterior i central des de ~0,4–0,6 s, p = 0,002.

### Per què afegir IM com a tercera classe no empitjora C vs IP

La tasca avaluada continua sent C vs IP. Els trials IM d'entrenament s'hi afegeixen com a tercera
classe i, al test (només C i IP), la seva probabilitat es descarta. Com que IM també és conflicte
(mateixa supressió d'alfa i beta), és en la pràctica més dades de conflicte; confondre IM amb IP no
penalitza. El guany (+1,9 punts sobre l'EEGNet conjunt) no és significatiu (p = 0,12).

### Identificació de subjectes: persona o sessió?

Cada participant té una sola sessió, així que persona i sessió no es poden separar. Per bandes
(divisió cronològica dins la sessió, atzar 0,021): delta 0,35, theta 0,50, alfa 0,70, beta 0,80–0,93,
gamma 0,95–0,96, totes 0,98. Les freqüències altes (soroll muscular i d'elèctrode) hi pesen molt.

### Línia base de comportament per a C vs IP

Posició al bloc + historial: 51,0 % (p = 0,17), conjunt 51,1 %. P(C | anterior C) = 0,368 contra
0,361 (n.s.). L'ordre dels trials és aleatori, així que el C vs IP no ve del comportament.
