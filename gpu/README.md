# Experiments de GPU a Kaggle (pas a pas)

Els models profunds (EEGNet, EEGNeX, EEG-Inception, ShallowNet i **CBraMod preentrenat**)
s'entrenen amb **tots els subjectes alhora**. Cada subjecte s'alinea amb Euclidean
Alignment, i després el model s'ajusta a cada subjecte. En CPU això trigaria dies;
en una GPU gratuïta de Kaggle, unes 5–7 hores per a les dues tasques.

El protocol és el mateix que el del benchmark local: el 20 % final de cada subjecte no
es toca fins a la puntuació final.

## 1. Preparar les dades (al teu ordinador, ~2 min)

```powershell
$env:PYTHONPATH = "src"; py -3.13 -m br_analysis.research export
```

Això crea `artifacts/research/_export/` amb quatre fitxers:

| Fitxer | Mida aprox. |
|---|---|
| `c_vs_ip.npz` | ~590 MB |
| `im_vs_ip.npz` | ~640 MB |
| `deep_pooled.py` | l'script |
| `kaggle_notebook.ipynb` | el notebook |

Les dades són èpoques ja preprocessades de la finestra d'estímul (0–1,5 s, 0,1–30 Hz,
250 Hz). No inclouen els FIF originals ni cap dada identificativa més enllà del codi de
subjecte.

## 2. Pujar-les a Kaggle com a dataset privat (~10 min, un sol cop)

1. Entra a <https://www.kaggle.com/datasets> → **New Dataset**.
2. Arrossega els fitxers `c_vs_ip.npz`, `im_vs_ip.npz` i `deep_pooled.py`.
3. Títol: per exemple `br-eeg-stimulus`. Visibilitat: **Private**. → **Create**.

Si cal, verifica el compte amb el telèfon: Kaggle ho demana per poder fer servir GPU.

## 3. Crear i llançar el notebook (~2 min)

1. <https://www.kaggle.com/code> → **New Notebook** → menú **File → Import Notebook** →
   puja `kaggle_notebook.ipynb`.
2. Al panell de la dreta:
   - **Session options → Accelerator: GPU T4 x2**.
   - **Internet: On**. Cal per descarregar braindecode i els pesos de CBraMod.
3. **Add Input** → *Your Datasets* → `br-eeg-stimulus`.
4. A dalt a la dreta: **Save Version → Save & Run All (Commit) → Save**.

Ja pots tancar el navegador: s'executa sol en segon pla. Kaggle et dona unes 30 h de GPU
setmanals i com a màxim 12 h per execució.

## 4. Recollir els resultats

1. Quan la versió surti com a *Complete*, obre-la → pestanya **Output** → descarrega
   `results.zip`.
2. Descomprimeix-lo, per exemple a `artifacts/research/_gpu/`, i importa'l:

```powershell
$env:PYTHONPATH = "src"; py -3.13 -m br_analysis.research import-gpu artifacts/research/_gpu/results
```

```powershell
$env:PYTHONPATH = "src"; py -3.13 -m br_analysis.research summary
```

Els models de GPU apareixen a la mateixa taula que els clàssics, amb les mateixes
estadístiques: permutacions per subjecte, prevalença i guany respecte a Riemann.

## Si alguna cosa falla

- **"No GPU"**: comprova que l'accelerador sigui GPU i que el compte estigui verificat.
- **Un model falla**: l'script continua amb els altres i escriu `FAILED: ...` al log.
  Envia'm el log de la pestanya *Logs* i ho arreglo.
- **Prova ràpida abans de llançar-ho tot**: canvia l'última línia de codi del notebook
  per `--suite --max-subjects 3 --epochs 2 --seeds 1`. Triga uns 5 minuts.
