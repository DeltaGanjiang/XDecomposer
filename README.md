# XDecomposer

XDecomposer separates phases in an experimental X-ray diffraction (XRD) pattern and matches each predicted phase to candidate Materials Project IDs (MP-IDs) from a local structure database.

## Important paths

| Path | Purpose |
| --- | --- |
| `exp_data/intensity.csv` | Input experimental XRD data. It is a headerless CSV with two columns: `2theta` (degrees) and intensity. |
| `exp_data_to_mpid.ipynb` | English Jupyter notebook that loads `exp_data/intensity.csv` and produces MP-ID predictions. |
| `exp_data/mpid_predictions.csv` | Best MP-ID prediction for each model phase; created when the notebook is run. |
| `MP500.db` | Local ASE database containing the MP500 candidate crystal structures and MP-IDs. |
| ` checkpoints/sepration/latest.pt` | Trained XDecomposer checkpoint used for phase separation. The leading space and `sepration` spelling are part of the existing directory name. |
| ` checkpoints/pretrain/best_model.pt` | Pretrained XRD encoder checkpoint required by the decomposition model. |
| `api/main.py` | FastAPI service and local inference implementation. It also defines MP-ID matching against `MP500.db`. |
| `api/README.md` | API startup and endpoint documentation. |
| `api/static/index.html` | Browser interface for uploading XRD data and viewing results. |
| `src/models/xdecomposer.py` | XDecomposer model implementation. |
| `src/models/xrd_transformer.py` | XRD Transformer encoder implementation. |
| `src/data/` | Dataset classes and data-loading utilities. |
| `datasets/` | Bundled MP500, RRUFF, and OPXRD datasets, patterns, manifests, and structures. |
| `scripts/python_runners/` | Python entry points for training, fine-tuning, and testing. |
| `scripts/bash_train/` | Shell scripts for training and metric aggregation. |
| `scripts/bash_test/` | Shell scripts for evaluation. |
| `configs/` | Training and ablation configuration files. |
| `requirements.txt` | Python dependencies. |

## Run the experimental-data notebook

From the repository root, start Jupyter and open the notebook:

```bash
jupyter notebook exp_data_to_mpid.ipynb
```

Run all cells. The notebook uses the local checkpoints and `MP500.db`, then shows the predicted active MP-ID(s) and writes `exp_data/mpid_predictions.csv`.

## Run the web API

The API cannot start with Python dependencies alone. Before starting it, make
sure these model resources are available on your machine:

| Required resource | Default location used by the API |
| --- | --- |
| XDecomposer checkpoint | ` checkpoints/sepration/latest.pt` |
| Pretrained XRD encoder checkpoint | ` checkpoints/pretrain/best_model.pt` |
| MP500 structures database | `MP500.db` |

The leading space in ` checkpoints` and the spelling `sepration` are
intentional: they match the paths currently defined in `api/main.py`. The
checkpoint files and database are not included in a clone that does not
contain those paths. Obtain them from the project maintainer or place your
own compatible files at those locations.

If your files are stored elsewhere, pass their absolute paths as environment
variables instead. This is usually less error-prone than creating a directory
whose name starts with a space:

```bash
python -m pip install fastapi "uvicorn[standard]"

XDECOMPOSER_CHECKPOINT="/absolute/path/latest.pt" \
XDECOMPOSER_MAE_CHECKPOINT="/absolute/path/best_model.pt" \
XDECOMPOSER_MATERIALS_DB="/absolute/path/MP500.db" \
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Replace all three `/absolute/path/...` values. Starting with `python -m`
ensures the installer and Uvicorn use the same Python environment. If a
resource is missing, startup stops with `Required model resource not found:`
followed by the exact path to correct.

Open `http://127.0.0.1:8000/` in a browser. See `api/README.md` for API request examples.
