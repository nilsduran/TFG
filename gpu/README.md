# Experiments de GPU a Kaggle (pas a pas)

Hi ha dos notebooks. Tots dos fan servir el mateix dataset privat de Kaggle (`c_vs_ip.npz`,
`im_vs_ip.npz`):

| Notebook | Què fa | Durada |
|---|---|---|
| `kaggle_screen.ipynb` | **Cribratge** ("spaghetti at the wall"): ~40 configuracions per tasca, amb eliminació progressiva i una GPU per tasca | 6–10 h |
| `kaggle_notebook.ipynb` | Suite fixa de 5 models (la primera tanda, ja feta) | 3–4 h |

El protocol és el mateix que el del benchmark local: el 20 % final de cada subjecte no es
toca fins a la puntuació final. Al cribratge, les rondes eliminatòries es decideixen només
amb trials de validació; el holdout només es puntua per als 3 finalistes de cada tasca i el
seu ensemble.

## Cribratge (`kaggle_screen.ipynb`)

### Què prova

Partint d'EEGNet + EA + augmentació, canvia **una cosa cada vegada**:

- **Arquitectures:** EEG-Inception, EEGNeX, ShallowNet, ATCNet, EEGConformer, Deep4,
  EEGSimpleConv, EEG-TCNet, EEG-ITNet, TIDNet, SCCNet, CTNet, MSVTNet, IFNet, FBCNet i
  TSception.
- **Filtres i resolució:** passa-alt 0,1 / 0,5 / 2 Hz, passa-baix 15 Hz, 64 i 250 Hz.
- **Finestres dins l'estímul:** 0–0,5 s, 0,15–0,65 s, 0–1 s, 0,2–1 s i 0,5–1,5 s.
- **Espai:** només canals posteriors, i sense Euclidean Alignment.
- **Regularització:** sense augmentació, augmentació forta, mixup, label smoothing i
  weight decay.
- **Optimització:** learning rate i mida del batch.
- **Multitasca:** la classe que li falta a cada tasca (C per a IM vs IP, IM per a C vs IP)
  s'afegeix com a tercera classe d'entrenament.

### Com elimina configuracions

1. **Ronda 1:** totes les configuracions, amb 1 llavor.
2. **Combinacions:** les 2 millors arquitectures més totes les millores que han superat la
   base per més de 0,01 d'AUC.
3. **Ronda 2:** les 6 millors fan 2 llavors més.
4. **Ronda 3:** les 3 millors passen el protocol complet (3 llavors, fine-tuning per
   subjecte i holdout). També es puntua l'ensemble de les tres.

Després de cada execució s'escriu una classificació (`_screen_leaderboard.csv`). Si
Kaggle tallés l'execució, no es perdria el que ja estigués fet.

### Passos (~5 min de feina teva)

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** → puja
   `gpu/kaggle_screen.ipynb`.
2. Al panell de la dreta:
   - **Accelerator: GPU T4 x2**. Les dues GPU es fan servir alhora, una per tasca.
   - **Internet: On**.
3. **Add Input** → *Your Datasets* → el mateix dataset d'abans (`br-eeg-stimulus`). No cal
   tornar-lo a pujar: l'script ja va dins el notebook.
4. **Save Version → Save & Run All (Commit) → Save**. Ja pots tancar el navegador.
5. Quan surti *Complete*: **Output** → descarrega `results.zip` → descomprimeix-lo a
   `artifacts/research/_gpu2/` i digues-m'ho.

Si abans vols una prova ràpida d'uns 10 minuts, posa `QUICK = True` a la cel·la de
llançament.

## Importar resultats (ho faig jo, però per si de cas)

```powershell
$env:PYTHONPATH = "src"; py -3.13 -m br_analysis.research import-gpu artifacts/research/_gpu2/results
```

```powershell
$env:PYTHONPATH = "src"; py -3.13 -m br_analysis.research summary
```

## Regenerar els fitxers

Si es modifica `deep_pooled.py`, cal regenerar el notebook amb
`py -3.13 gpu/make_notebook.py`. També es regenera automàticament quan es fa `export`.

## Si alguna cosa falla

- **"No GPU"**: comprova que l'accelerador sigui GPU i que el compte estigui verificat amb
  el telèfon.
- **Una configuració falla**: el cribratge continua amb la resta i ho apunta com a
  `FAILED` a la classificació.
- **Logs**: `results/screen_c_vs_ip.log` i `results/screen_im_vs_ip.log` van dins el zip.
