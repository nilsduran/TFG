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

## 3. Següent pas: cribratge ampli a GPU

`gpu/kaggle_screen.ipynb` prova unes 40 configuracions per tasca (arquitectures, filtres,
finestres, canals, alineament, augmentació, optimització i multitasca) amb eliminació
progressiva:

- Les rondes eliminatòries fan servir només validació, que durant el cribratge s'amplia
  al 25 % final del desenvolupament.
- Les combinacions de millores es generen automàticament.
- Els 3 finalistes de cada tasca, i el seu ensemble, passen el protocol complet amb
  holdout.

Les instruccions són a `gpu/README.md`.

## 4. Pendent

- Entrenament conjunt clàssic (Riemann i ERP-LDA amb EA), en curs en local. Dirà quina
  part del guany ve de l'entrenament conjunt i quina de la xarxa neuronal.
- Importar i analitzar el cribratge: prevalença, comparació aparellada i, si cal, una
  segona ronda centrada en el que funcioni.
- Si cap direcció mou IM vs IP, documentar-ho com a resultat negatiu robust: s'han provat
  més de 40 configuracions i 15 arquitectures.
