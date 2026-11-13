# Replication package — Continuous structural complexity of history-based parametric CAD models and its association with parametric robustness

This package contains the metrics, perceived-complexity scores (S_MAP), analysis code and results needed to reproduce every table, figure and number of the manuscript. The native CAD files are **not** included (see "What is not included").

## Contents

| Path | Description |
|---|---|
| `data/Dataset_complejidad_con_size.xlsx` | Metrics of the 351 analysed models (352 records; one model was extracted twice) as exported by QualiCAD. This is the file read by the code. |
| `data/dataset_metrics_352_records.csv` | Same data in CSV, with the target column named `success_ratio` (it is `succes_ratio` in the xlsx). |
| `data/resultados_106.xlsx` | Perceived-complexity study: evaluator orderings (`resultados_no`), scales (`escalas`, `solver original` → `s_map`) and correlation sheets. |
| `data/smap_scores_106.csv` | The `solver original` sheet (S_MAP and alternative scales) for the 106 evaluated models. |
| `data/model_identifiers.csv` | File name, numeric model id and source of each model. |
| `code/` | Python analysis code (`pipeline.py`, `additional_analyses.py`, `export_reference_transformation.py`, `hoeffding.py`, `config.py`, `figura_flujo.py`). |
| `results/` | Outputs of the code: `tablas.xlsx` (tables), `valores_paper.json` (numbers quoted in the text), `additional_analyses.json` (sensitivity analyses, bootstrap and agreement statistics), `dataset_procesado.xlsx` (ICE per model), `ice_reference_transformation.json` (reference transformation of ICE), `figures/`. |
| `SHA256SUMS.txt` | Checksums of all files. |

## How to run

```
pip install -r code/requirements.txt
cd code
python pipeline.py                         # tables, figures and numbers (about 30 s)
python additional_analyses.py              # sensitivity analyses, bootstrap, Kendall's W, algebraic recovery of N and E
python export_reference_transformation.py  # fitted quantile transform, scaler and PC1 loadings that define ICE
```

Python 3.10+; the seed is fixed (`config.SEMILLA = 20260905`), so repeated runs give identical figures. Results were checked to be identical to those of the manuscript analysis; a run on 2026-11-13 used scikit-learn 1.9.1.

## Computing ICE for new models

ICE (Structural Complexity Index; from the Spanish *Índice de Complejidad Estructural*) is the first principal component of the 10 retained metrics after a quantile→normal transformation and standardisation. It is defined **relative to this 351-model reference corpus**. `results/ice_reference_transformation.json` stores the empirical quantiles, scaler and loadings needed to apply the same transformation to new models (metric order in `variables`).

## Data notes

- **Variables.** Nineteen candidate metrics plus the target were extracted with QualiCAD through the SolidWorks API; ten are retained in the study (see `code/config.py`: three are unusable because of an extraction failure, five are redundant, `time` is excluded).
- **`time`** is the time recorded by QualiCAD during metric extraction. It is not the authoring time of the model (it differs between the two extractions of the duplicated file).
- **`success_ratio`** is the proportion of the tested dimensional modifications that regenerate without error, computed by QualiCAD for each file. Its relation to the REM metric of the reference study could not be verified.
- **`entropy_shannon`** takes only rational values k/n (denominator not larger than the number of nodes recovered for the model), so it behaves as a ratio of distinct to total elements rather than as a logarithmic entropy; **`entropy_infoshannon`** is continuous (0.95–1.00). The exact QualiCAD definition of both columns could not be verified. `entropy_wang` is the entropy of Li et al., called Li entropy in the manuscript.
- **Source labels.** `ExercisesBook` and `ModelosLibroPedro` appear as `Textbook-A` and `Textbook-B`, and `SPECap_*` as `SPECapc_*`, as in the manuscript.
- `n_dimensions`, `is_surface_model` and `n_dependencies` are corrupted by an export failure and are not used.

## What is not included

- Native `.sldprt` models. GrabCAD models cannot be redistributed; `data/model_identifiers.csv` lists the file name and numeric id of every model so the corpus can be reassembled from its public sources.
- The QualiCAD source code.

## License

Code: MIT (`LICENSE`). Data and results: CC BY 4.0 (`LICENSE-DATA.md`).
