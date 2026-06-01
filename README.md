# Churn Prediction Pipeline

An end-to-end ML system: data ingestion → feature engineering → model training → REST API → monitoring dashboard.

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | **3.11** (exact) | 3.10 and below are not compatible; 3.12+ untested |
| pip | 24+ | bundled with Python 3.11 |
| Git | any | to clone this repo |
| Internet access | — | to download the Telco dataset (~300 KB from GitHub) |
| **Docker** + **Docker Compose** | 24+ / 2.x | optional — only needed for the containerised path |

> **Why Python 3.11 exactly?**  `xgboost==2.0.3` and `imbalanced-learn==0.12.3` have binary wheels only for 3.11 on most platforms. The `Dockerfile` uses `python:3.11-slim` for the same reason.

---

## Setup (local, recommended)

### 1 — Clone and create a virtual environment

```bash
git clone <repo-url>
cd customer-prediction-project

# Create an isolated environment (keeps your system Python clean)
python3.11 -m venv .venv

# Activate it
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
```

### 2 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs (among others):
- `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost` — data + modelling
- `fastapi`, `uvicorn`, `pydantic` — REST API
- `streamlit`, `evidently`, `plotly` — dashboard + drift detection
- `shap` — feature explanations
- `pytest` — tests

> **Note:** `shap` requires a C++ compiler on some platforms (`gcc`/`build-essential` on Linux, Xcode CLI on macOS, Visual C++ Build Tools on Windows). If installation fails, install the compiler first or use the Docker path below.

### 3 — Run the training pipeline

```bash
python run_pipeline.py
```

This executes the three training steps in order:

| Step | Script | Output |
|---|---|---|
| Data ingestion | `src/data/ingest.py` | `data/raw/telco.csv` |
| Feature engineering | `src/features/build.py` | `data/processed/features.csv`, `models/pipeline.pkl` |
| Model training | `src/models/train.py` | `models/churn_model.pkl` |

**Flags:**

```bash
# Skip the download if you already have data/raw/telco.csv
python run_pipeline.py --skip-ingest

# Skip feature engineering if data/processed/features.csv already exists
python run_pipeline.py --skip-features

# Train the model AND immediately start the API + dashboard
python run_pipeline.py --serve
```

### 4 — Start the services (separately)

Open two terminals (both with the venv active and run from the project root):

```bash
# Terminal 1 — REST API on http://localhost:8000
uvicorn src.api.main:app --reload

# Terminal 2 — Streamlit dashboard on http://localhost:8501
streamlit run dashboard/app.py
```

| URL | What |
|---|---|
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/metrics` | Prediction counter |
| `http://localhost:8000/predict` | `POST` — single prediction |
| `http://localhost:8000/predict/batch` | `POST` — batch predictions |
| `http://localhost:8501` | Streamlit monitoring dashboard |

### 5 — Run tests

```bash
pytest tests/ -v
```

---

## Setup (Docker)

The Docker path skips all local Python/compiler setup but **requires the model to be trained first** — the containers mount the `models/` directory from your host.

```bash
# Step 1: train the model locally (needs Python 3.11 + deps)
python run_pipeline.py

# Step 2: start API + dashboard containers
docker-compose up --build
```

Services are available at the same ports (8000 and 8501).

---

## Project structure

```
customer-prediction-project/
├── data/
│   ├── raw/              # Original dataset — never modified (created by ingest.py)
│   └── processed/        # Feature-engineered CSV (created by build.py)
├── models/               # Serialised artifacts (created by train pipeline)
│   ├── churn_model.pkl   # Trained XGBoost classifier
│   └── pipeline.pkl      # sklearn ColumnTransformer (preprocessing)
├── src/
│   ├── data/
│   │   └── ingest.py     # Downloads & validates raw Telco CSV
│   ├── features/
│   │   └── build.py      # Preprocessing pipeline + SMOTE
│   ├── models/
│   │   ├── train.py      # Trains XGBoost, logs metrics, saves artifact
│   │   └── predict.py    # Loads artifacts, runs inference
│   ├── api/
│   │   └── main.py       # FastAPI: /predict, /predict/batch, /health, /metrics
│   └── monitoring/       # (placeholder) drift detection
├── dashboard/
│   └── app.py            # Streamlit monitoring dashboard
├── tests/
│   └── test_features.py  # Unit tests for feature engineering
├── docker/
│   └── Dockerfile        # python:3.11-slim image
├── docker-compose.yml    # API + dashboard services
├── config.yaml           # All paths, feature lists, model hyperparameters
├── requirements.txt      # Pinned dependencies
└── run_pipeline.py       # One-shot pipeline runner script
```

---

## Configuration

Everything is controlled by `config.yaml`:

```yaml
data:
  dataset_url: ...      # Source for the Telco CSV
  raw_path: ...         # Where to save the raw file
  processed_path: ...   # Where to save processed features

model:
  xgboost_params:       # n_estimators, max_depth, learning_rate, etc.
  artifact_path: ...    # Path to save the trained model
  pipeline_path: ...    # Path to save the preprocessing pipeline
```

You don't need to touch this file to run the project with default settings.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure your venv is activated and `pip install -r requirements.txt` completed without errors |
| `shap` fails to install | Install C++ build tools (see Setup step 2 note) or use Docker |
| `FileNotFoundError: models/churn_model.pkl` | Run `python run_pipeline.py` to train the model before starting the API |
| `FileNotFoundError: data/raw/telco.csv` | Check internet access; the file is downloaded from GitHub at runtime |
| `uvicorn` command not found | Your venv may not be activated, or install failed — re-run `pip install -r requirements.txt` |
| Port 8000 / 8501 already in use | Kill the existing process or change the port in `config.yaml` and the `docker-compose.yml` |

---

## Build plan

### Phase 1 — Data (Days 1–2)
- [x] Download Telco dataset via `ingest.py`
- [ ] Explore with `notebooks/01_exploration.ipynb`
- [ ] Document class imbalance, nulls, outliers

### Phase 2 — Features (Days 3–4)
- [x] Build preprocessing pipeline in `features/build.py`
- [x] Handle imbalance with SMOTE
- [x] Save processed dataset to `data/processed/`

### Phase 3 — Model (Days 5–6)
- [x] Train XGBoost in `models/train.py`
- [x] Log metrics: AUC, precision, recall, F1
- [x] Save model artifact with joblib

### Phase 4 — API (Days 7–8)
- [x] FastAPI endpoint with Pydantic validation
- [x] `/predict` returns churn probability + explanation
- [x] `/health` and `/metrics` endpoints

### Phase 5 — Monitoring (Days 9–10)
- [x] Streamlit dashboard showing live predictions
- [ ] Drift detection comparing new vs training distributions (evidently wired up)
- [ ] Alerts when model performance degrades

### Phase 6 — Deploy (Days 11–12)
- [x] Dockerize everything
- [x] docker-compose for API + dashboard
- [x] Write tests, CI-ready
