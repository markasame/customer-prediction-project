# Codebase Explanation — Churn Prediction Pipeline

A detailed walkthrough of every script, every function, every decorator, and how the pieces chain together.

---

## Table of Contents

1. [Overall Workflow](#1-overall-workflow)
2. [Configuration — `config.yaml`](#2-configuration--configyaml)
3. [Pipeline Runner — `run_pipeline.py`](#3-pipeline-runner--run_pipelinepy)
4. [Data Ingestion — `src/data/ingest.py`](#4-data-ingestion--srcdataingestpy)
5. [Feature Engineering — `src/features/build.py`](#5-feature-engineering--srcfeaturesbuildpy)
6. [Model Training — `src/models/train.py`](#6-model-training--srcmodelstrainpy)
7. [Prediction — `src/models/predict.py`](#7-prediction--srcmodelspredictpy)
8. [REST API — `src/api/main.py`](#8-rest-api--srcapimainpy)
9. [Dashboard — `dashboard/app.py`](#9-dashboard--dashboardapppy)
10. [Tests — `tests/test_features.py`](#10-tests--teststest_featurespy)
11. [Data Flow Summary](#11-data-flow-summary)

---

## 1. Overall Workflow

The pipeline has two distinct phases: **training** (one-shot, sequential) and **serving** (long-running, parallel).

```
TRAINING PHASE  (run once, in order)
─────────────────────────────────────────────────────────────────────────
  config.yaml
       │
       ▼
  ingest.py  ──────────────────────►  data/raw/telco.csv
       │
       ▼
  build.py   ──────────────────────►  data/processed/features.csv
             ──────────────────────►  models/pipeline.pkl
       │
       ▼
  train.py   ──────────────────────►  models/churn_model.pkl
       │
       ▼
  (training complete)


SERVING PHASE  (long-running, started after training)
─────────────────────────────────────────────────────────────────────────
  models/churn_model.pkl  ─┐
  models/pipeline.pkl     ─┤──► src/api/main.py  (FastAPI, port 8000)
  config.yaml             ─┘          │
                                       │  HTTP POST /predict
                                       ▼
                               dashboard/app.py  (Streamlit, port 8501)
```

**Trigger chain:**
- `run_pipeline.py` calls each training script as a subprocess in order.
- The API and dashboard are independent long-running processes — they don't call each other's Python code; the dashboard communicates with the API over HTTP.

---

## 2. Configuration — `config.yaml`

**Purpose:** Single source of truth for every path, feature list, and hyperparameter used across all scripts. Every script loads this file at startup with `yaml.safe_load()`.

```yaml
data:
  raw_path: data/raw/telco.csv          # where ingest.py saves the download
  processed_path: data/processed/features.csv  # where build.py saves features
  target_column: Churn                  # column name of the label in the CSV
  dataset_url: "https://..."            # source URL for the Telco dataset

features:
  numeric:                              # columns scaled with StandardScaler
    - tenure
    - MonthlyCharges
    - TotalCharges
  categorical:                          # columns encoded with OrdinalEncoder
    - gender
    - SeniorCitizen
    - ...
  drop:                                 # columns removed before any processing
    - customerID

model:
  artifact_path: models/churn_model.pkl   # saved XGBoost model
  pipeline_path: models/pipeline.pkl      # saved sklearn preprocessing pipeline
  test_size: 0.2                          # fraction of data held out for evaluation
  random_state: 42                        # seed for reproducibility everywhere
  xgboost_params:                         # passed directly to XGBClassifier(**params)
    n_estimators: 300
    max_depth: 5
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
    scale_pos_weight: 3                   # compensates for class imbalance (~3:1 ratio)
    eval_metric: auc

api:
  host: 0.0.0.0
  port: 8000
  model_path: models/churn_model.pkl
  pipeline_path: models/pipeline.pkl

monitoring:
  reference_data_path: data/processed/features.csv
  drift_threshold: 0.15                  # PSI score above which drift is flagged
```

> All scripts use `CONFIG_PATH = "config.yaml"` (a relative path), so they **must be run from the project root**.

---

## 3. Pipeline Runner — `run_pipeline.py`

**Purpose:** Orchestrates the three training scripts as subprocesses, with optional flags to skip already-completed steps and to launch services after training.

**How to run:**
```bash
python run_pipeline.py
python run_pipeline.py --skip-ingest --serve
```

### Constants

```python
MIN_PYTHON = (3, 11)
```
Minimum Python version tuple. Compared against `sys.version_info` at startup.

```python
PIPELINE_STEPS = [
    ("Data ingestion",      "src/data/ingest.py"),
    ("Feature engineering", "src/features/build.py"),
    ("Model training",      "src/models/train.py"),
]
```
Ordered list of `(label, script_path)` tuples that defines the pipeline execution order.

```python
ROOT = os.path.dirname(os.path.abspath(__file__))
```
Absolute path to the project root. Used as the `cwd` argument for every subprocess so that relative paths in the scripts (`config.yaml`, `data/`, `models/`) resolve correctly regardless of where the user runs this script from.

---

### Functions

#### `banner(text: str) -> None`
Prints a 62-character `===` border with `text` inside. Pure cosmetic — makes step transitions visible in terminal output.

---

#### `check_python() -> None`
Reads `sys.version_info` and calls `sys.exit()` with a human-readable message if the current Python is older than `MIN_PYTHON`. Prevents cryptic errors from incompatible pandas/xgboost wheel versions.

---

#### `run_step(label: str, script: str) -> None`
Runs a single training script as a blocking subprocess.

- `subprocess.run([sys.executable, script], cwd=ROOT)` — uses the same Python executable as the runner itself (important inside a venv), passes `ROOT` as working directory.
- If `result.returncode != 0`, calls `sys.exit(result.returncode)` to abort the whole pipeline and surface the correct exit code to the caller (CI systems, shell scripts, etc.).

---

#### `serve() -> None`
Starts the API and dashboard after training completes. Called only when `--serve` is passed.

1. Launches `uvicorn src.api.main:app` with `subprocess.Popen` — **non-blocking**, runs in the background.
2. Sleeps 3 seconds to give uvicorn time to bind the port before the dashboard tries to connect to it.
3. Launches `streamlit run dashboard/app.py` with `subprocess.run` — **blocking**, keeps the terminal attached so the user sees Streamlit output.
4. A `try/finally` block ensures `api_proc.terminate()` + `api_proc.wait()` are always called when the user hits Ctrl+C, so the uvicorn process doesn't become an orphan.

---

#### `main() -> None`
Entry point. Three responsibilities:

1. **Argument parsing** via `argparse.ArgumentParser`:
   - `--skip-ingest` — adds `"src/data/ingest.py"` to a `skip` set.
   - `--skip-features` — adds `"src/features/build.py"` to `skip`.
   - `--serve` — sets a boolean to call `serve()` at the end.

2. **Pipeline execution** — iterates `PIPELINE_STEPS`, skips entries in `skip`, calls `run_step()` for the rest. Times the full run with `time.time()`.

3. **Post-run output** — prints artifact paths if not serving, or calls `serve()` if the flag was set.

---

## 4. Data Ingestion — `src/data/ingest.py`

**Purpose:** Downloads the raw IBM Telco churn CSV from GitHub and validates that it has the expected shape and target column.

**Reads:** nothing (downloads from the internet)
**Writes:** `data/raw/telco.csv`

**How to run:**
```bash
python src/data/ingest.py
```

---

### Functions

#### `load_config() -> dict`
Opens `config.yaml` and returns its contents as a Python dictionary using `yaml.safe_load()`. Used by every script in the project — it is not a shared utility but each script has its own copy of this one-liner.

---

#### `download_dataset(url: str, dest: str) -> None`
Downloads a file from `url` and saves it to `dest`.

- `os.makedirs(os.path.dirname(dest), exist_ok=True)` — creates `data/raw/` if it does not exist yet. `exist_ok=True` means no error if it already exists.
- `requests.get(url, timeout=30)` — makes a GET request with a 30-second timeout. The timeout prevents the script hanging indefinitely on a slow connection.
- `response.raise_for_status()` — raises `requests.HTTPError` for 4xx/5xx responses, so a 404 or 403 fails loudly rather than saving an error page as a CSV.
- Writes in binary mode (`"wb"`) to avoid any line-ending translation.

---

#### `validate_dataset(path: str, target_col: str) -> pd.DataFrame`
Loads the saved CSV and checks it is usable.

- Prints shape, column names, and any columns that have null values.
- `assert target_col in df.columns` — hard fails if the expected label column (`Churn`) is missing. This is an intentional crash: it means the download was corrupted or the dataset changed.
- Prints the class distribution of the target column. If `churn_rate < 0.3`, it notes that class imbalance is present (Telco's churn rate is ~26%, which is why SMOTE is applied later).
- Returns the DataFrame so `main()` can report the row count.

---

#### `main() -> None`
Orchestration function, called when the script is run directly.

1. Loads config.
2. Checks if `data/raw/telco.csv` already exists with `os.path.exists()` — skips the download if so (idempotent behaviour).
3. Calls `validate_dataset()` and prints the row count.

---

## 5. Feature Engineering — `src/features/build.py`

**Purpose:** Cleans the raw data, builds a reusable sklearn preprocessing pipeline, applies SMOTE to balance classes, and saves both the processed feature matrix and the fitted pipeline artifact.

**Reads:** `data/raw/telco.csv`, `config.yaml`
**Writes:** `data/processed/features.csv`, `models/pipeline.pkl`

**How to run:**
```bash
python src/features/build.py
```

---

### Imports worth noting

```python
from imblearn.pipeline import Pipeline as ImbPipeline
```
`imbalanced-learn` provides a drop-in replacement for sklearn's `Pipeline` that correctly handles SMOTE — standard sklearn Pipeline does not support resamplers. Imported but not actually used in the current implementation (SMOTE is applied manually instead); included for future refactoring.

---

### Functions

#### `load_config() -> dict`
Same pattern as in `ingest.py` — loads and returns `config.yaml`.

---

#### `clean_raw_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]`
Prepares the raw DataFrame for the sklearn pipeline.

- `df.drop(columns=cfg["features"]["drop"], errors="ignore")` — removes `customerID`. `errors="ignore"` means no crash if the column is already absent.
- `pd.to_numeric(df["TotalCharges"], errors="coerce")` — the Telco dataset stores `TotalCharges` as a string. New customers (zero tenure) have a single space `" "` instead of a number. `errors="coerce"` turns those spaces into `NaN`.
- `df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])` — fills the `NaN`s with `MonthlyCharges` (a reasonable proxy: if they've been a customer for 0 months, their total equals one month's charge). Written as an assignment (not `inplace=True`) to comply with pandas Copy-on-Write semantics introduced in pandas 2.0.
- `(df[target_col].str.strip() == "Yes").astype(int)` — converts the `"Yes"/"No"` strings to `1/0` integers. `.str.strip()` guards against whitespace in the source CSV.
- Returns `(X, y)` — features DataFrame and target Series separately.

---

#### `build_preprocessing_pipeline(cfg: dict) -> ColumnTransformer`
Constructs (but does not fit) the sklearn preprocessing pipeline.

**Numeric sub-pipeline** (applied to `tenure`, `MonthlyCharges`, `TotalCharges`):
```
SimpleImputer(strategy="median")  →  StandardScaler()
```
- `SimpleImputer` fills any remaining `NaN`s with the column median (robust to outliers).
- `StandardScaler` zero-centres and scales to unit variance — required for distance-based algorithms and generally beneficial for tree-based ones with regularisation.

**Categorical sub-pipeline** (applied to `gender`, `Contract`, etc.):
```
SimpleImputer(strategy="most_frequent")  →  OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
```
- `SimpleImputer` fills missing strings with the most common value.
- `OrdinalEncoder` assigns an integer to each category. `handle_unknown="use_encoded_value"` with `unknown_value=-1` means unseen categories at inference time get `-1` instead of raising an error — critical for a production API that might receive novel values.

**ColumnTransformer** combines both sub-pipelines:
- `remainder="drop"` — any columns not listed in `numeric` or `categorical` are silently dropped. This means future schema additions to the raw data won't break the pipeline.

Returns the unfitted `ColumnTransformer` object.

---

#### `save_features(X_resampled, y_resampled, path: str, cfg: dict) -> None`
Saves the post-SMOTE feature matrix as a CSV for reference and later use by the monitoring dashboard.

- Reconstructs column names from `cfg["features"]["numeric"] + cfg["features"]["categorical"]`.
- Appends the target column `Churn` as the last column.
- `os.makedirs(os.path.dirname(path), exist_ok=True)` — creates `data/processed/` if needed.

---

#### `main() -> None`
Orchestration function.

1. Loads raw CSV and calls `clean_raw_data()`.
2. Calls `build_preprocessing_pipeline()` and immediately fits it: `preprocessor.fit_transform(X)`. This both fits the transformers (learns mean, std, category mappings) and transforms the data in one pass.
3. Applies SMOTE: `SMOTE(random_state=...).fit_resample(X_processed, y)`. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples of the minority class (churned customers) by interpolating between existing minority samples in feature space. After SMOTE, the class ratio is 1:1.
4. Saves the fitted `preprocessor` with `joblib.dump()` — this serialised object is what the API loads at startup to transform incoming customer data before feeding it to the model.
5. Calls `save_features()`.

---

## 6. Model Training — `src/models/train.py`

**Purpose:** Loads the processed features, trains an XGBoost classifier, evaluates it on a held-out test set, runs cross-validation, and saves the model artifact.

**Reads:** `data/processed/features.csv`, `config.yaml`
**Writes:** `models/churn_model.pkl`

**How to run:**
```bash
python src/models/train.py
```

---

### Functions

#### `load_config() -> dict`
Same pattern — loads `config.yaml`.

---

#### `load_processed_data(path: str) -> tuple[np.ndarray, np.ndarray]`
Reads the processed CSV and splits it into numpy arrays.

- The last column (`Churn`) is the target `y`; everything else is `X`.
- Returns numpy arrays (not DataFrames) because XGBoost and sklearn metrics work with arrays and avoid DataFrame overhead during training.

---

#### `train_model(X_train, y_train, params: dict) -> XGBClassifier`
Creates and fits the XGBoost model.

- `XGBClassifier(**params, use_label_encoder=False, verbosity=0)` — unpacks all hyperparameters from `config.yaml` directly. `use_label_encoder=False` suppresses a deprecation warning; `verbosity=0` silences XGBoost's own output (evaluation is done separately).
- `model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)` — trains the model. `eval_set` is required to activate `eval_metric: auc` in the params, even though the result is not printed here.
- Returns the fitted `XGBClassifier`.

---

#### `evaluate_model(model, X_test, y_test) -> dict`
Evaluates the trained model on the held-out test set and prints a full report.

- `model.predict(X_test)` — returns hard class labels (0 or 1).
- `model.predict_proba(X_test)[:, 1]` — returns the probability of class 1 (churn). `[:, 1]` selects the second column (the positive class).
- **ROC-AUC** (`roc_auc_score`) — area under the Receiver Operating Characteristic curve. Threshold-independent measure of ranking quality; 0.5 is random, 1.0 is perfect.
- **Average Precision** (`average_precision_score`) — area under the Precision-Recall curve. More informative than AUC on imbalanced datasets because it focuses on the minority class.
- `classification_report` — prints precision, recall, F1, and support for both classes.
- `confusion_matrix` — prints the 2×2 table of TP/FP/FN/TN.
- Returns a `dict` of numeric metrics (not used further but available for callers to log).

---

#### `cross_validate_model(model, X, y, n_splits=5) -> None`
Runs stratified k-fold cross-validation on the full dataset.

- `StratifiedKFold` preserves the class ratio in every fold — essential for imbalanced data.
- `cross_val_score(..., scoring="roc_auc")` — returns an array of AUC scores, one per fold.
- Prints mean ± std. High std (> 0.02) suggests the model is sensitive to the data split.

---

#### `save_model(model, path: str) -> None`
Serialises the fitted model to disk with `joblib.dump()`.

- `os.makedirs(os.path.dirname(path), exist_ok=True)` — creates `models/` if needed.
- `joblib` is preferred over `pickle` for sklearn/XGBoost objects because it handles large numpy arrays more efficiently.

---

#### `main() -> None`
Orchestration function.

1. Loads processed data.
2. `train_test_split(..., stratify=y)` — stratified split ensures the held-out test set has the same churn ratio as the training set.
3. Trains, evaluates, cross-validates, and saves.

---

## 7. Prediction — `src/models/predict.py`

**Purpose:** Utility module (not run directly). Provides `load_artifacts()` and `predict()` to the API. Centralises inference logic so the API doesn't contain any ML code.

---

### Functions

#### `load_artifacts(config_path: str = "config.yaml") -> tuple[XGBClassifier, ColumnTransformer, dict]`
Loads both serialised artifacts from disk and returns them together with the config.

- Called once at API startup (not on every request) for performance.
- Returns the model, the preprocessing pipeline, and the full config dict.

---

#### `predict(customer_data: dict, model, pipeline, cfg) -> dict`
Runs a single inference.

1. `pd.DataFrame([customer_data])` — wraps the input dict in a single-row DataFrame. The preprocessing pipeline expects a DataFrame with named columns.
2. `pipeline.transform(df)` — applies the same transformations that were fitted during `build.py`. Note: `.transform()` not `.fit_transform()` — the pipeline must not be re-fitted on new data.
3. `model.predict_proba(X)[0, 1]` — extracts the churn probability for the single row.
4. **Risk label** thresholds:
   - `prob >= 0.7` → `"High"`
   - `0.4 <= prob < 0.7` → `"Medium"`
   - `prob < 0.4` → `"Low"`
5. **Top risk factors** — zips feature names with `model.feature_importances_` (a property of the fitted XGBClassifier that returns per-feature gain scores), sorts descending, and returns the top 5 feature names. These are global importances from training, not per-prediction SHAP values — a simplification that avoids the runtime cost of SHAP.
6. Returns a dict with `churn_probability`, `risk_label`, and `top_risk_factors`.

---

## 8. REST API — `src/api/main.py`

**Purpose:** Exposes the trained model as an HTTP service using FastAPI. Handles input validation, runs inference, and tracks basic metrics.

**Reads:** `models/churn_model.pkl`, `models/pipeline.pkl`, `config.yaml` (at startup)
**Writes:** nothing

**How to run:**
```bash
uvicorn src.api.main:app --reload
# or
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

### Module-level setup (runs at import time)

```python
app = FastAPI(title="Churn Prediction API", ...)
```
Creates the FastAPI application instance. This is what uvicorn receives and serves.

```python
model, pipeline, cfg = load_artifacts()
```
Loads both model artifacts **once** when the module is imported (i.e., at server startup). If the model files don't exist, the server fails here with a `FileNotFoundError` — intentional, since there is no point starting the API without a trained model.

```python
prediction_count = 0
start_time = time.time()
```
Simple in-memory counters for the `/metrics` endpoint. Reset every time the server restarts.

---

### Pydantic models (data schemas)

#### `CustomerFeatures(BaseModel)`
Input validation schema for a single prediction request.

- Every field is a Python type annotation. FastAPI uses Pydantic to:
  1. Parse the incoming JSON body.
  2. Validate types and constraints (e.g., `ge=0` on `tenure` rejects negative values).
  3. Return a `422 Unprocessable Entity` response automatically if validation fails.
- `Field(example=...)` — populates the Swagger UI at `/docs` with example values so the API is immediately testable in a browser.

#### `PredictionResponse(BaseModel)`
Output schema. FastAPI serialises the return value of route functions to JSON using this model. If the function returns a dict with extra keys, they are stripped. If required keys are missing, a `500` is raised.

---

### Route handlers (decorated functions)

#### `@app.get("/health")`
```python
def health():
```
**Decorator:** `@app.get("/health")` registers this function as the handler for `GET /health`. FastAPI adds the route to the app's router at import time.

Returns `{"status": "ok", "uptime_seconds": ...}`. Used by the Docker healthcheck (`curl -f http://localhost:8000/health`) and the Streamlit dashboard's status indicator.

---

#### `@app.get("/metrics")`
```python
def metrics():
```
**Decorator:** `@app.get("/metrics")` registers the handler for `GET /metrics`.

Returns the total prediction count and uptime. Intended for lightweight operational monitoring — not a replacement for Prometheus/Grafana.

---

#### `@app.post("/predict", response_model=PredictionResponse)`
```python
def predict_churn(customer: CustomerFeatures):
```
**Decorator:** `@app.post("/predict", response_model=PredictionResponse)` registers the handler for `POST /predict`.

- `response_model=PredictionResponse` tells FastAPI to validate and serialise the return value using `PredictionResponse`. Extra fields in the returned dict are silently dropped; missing required fields raise a 500.
- `customer: CustomerFeatures` — FastAPI reads the request body, validates it against `CustomerFeatures`, and passes a fully typed object. No manual JSON parsing needed.
- `customer.model_dump()` — converts the Pydantic object back to a plain dict to pass to `predict()`.
- `global prediction_count` — required to mutate the module-level counter inside a function.
- Wraps everything in `try/except` and raises `HTTPException(status_code=500)` on any error, so the server never returns an unstructured traceback to the client.
- Logs every prediction at INFO level: probability and risk label.

---

#### `@app.post("/predict/batch", response_model=list[PredictionResponse])`
```python
def predict_batch(customers: list[CustomerFeatures]):
```
**Decorator:** Same pattern as above but with `list[PredictionResponse]` as the response model.

- `customers: list[CustomerFeatures]` — FastAPI parses a JSON array of customer objects.
- Iterates the list, calls `predict()` for each, appends to `results`. If any single prediction fails, the whole batch fails with a 500 (no partial results).
- Each successful prediction increments `prediction_count`.

---

## 9. Dashboard — `dashboard/app.py`

**Purpose:** A Streamlit web app for monitoring the pipeline. Shows prediction distributions, risk breakdowns, a time-series chart, and a drift detection panel. Also provides a quick-predict form that calls the live API.

**Reads:** `config.yaml`, calls `http://localhost:8000` over HTTP
**Writes:** nothing

**How to run:**
```bash
streamlit run dashboard/app.py
```

---

### How Streamlit works

Streamlit re-executes the **entire script** from top to bottom on every user interaction (slider move, button click, page refresh). There is no explicit event loop — the framework handles re-runs. This means:
- Every function call that makes a network request or reads a file runs again on each interaction.
- `st.cache_data` / `st.cache_resource` can be used to memoize expensive calls (not used here).

---

### Module-level setup

```python
st.set_page_config(page_title="Churn Pipeline Monitor", page_icon="📉", layout="wide")
```
Must be the first Streamlit call in the script. Sets browser tab title, favicon, and uses the full browser width.

```python
API_URL = "http://localhost:8000"
```
Hardcoded API address. The dashboard assumes it is running on the same machine as the API.

---

### Functions

#### `load_config() -> dict`
Loads `config.yaml` — same pattern as other scripts.

---

#### `check_api_health() -> dict | None`
Makes a `GET /health` request to the API with a 3-second timeout.

- Returns the parsed JSON dict on success.
- Returns `None` on any exception (connection refused, timeout, non-200 status). Used to set the sidebar status indicator to green or red without crashing the dashboard if the API is offline.

---

#### `get_api_metrics() -> dict`
Makes a `GET /metrics` request to the API.

- Returns the JSON dict, or an empty dict `{}` on failure.
- The dashboard adds `api_metrics.get("total_predictions", 0)` to the demo count to show a combined total.

---

#### `generate_demo_predictions(n=200) -> pd.DataFrame`
Generates synthetic prediction history for demonstration purposes — the dashboard is functional before any real traffic has been served.

- `np.random.seed(42)` — fixed seed so the demo data is identical on every page load (no jitter).
- `np.random.beta(2, 5, n)` — generates probabilities skewed toward lower values (realistic: most customers do not churn).
- `pd.cut(probs, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])` — buckets probabilities into risk labels using the same thresholds as `predict.py`.
- Returns a DataFrame with columns: `timestamp`, `churn_probability`, `risk_label`, `tenure`, `MonthlyCharges`, `Contract`.

---

### Layout structure (module-level Streamlit calls)

Streamlit builds the page by executing these calls in order:

**Sidebar** (`with st.sidebar:` context manager)
- API health badge: `st.success()` or `st.error()` based on `check_api_health()`.
- Quick-predict sliders: `st.slider()` for tenure and monthly charges; `st.selectbox()` for contract type.
- "Run prediction →" button: `st.button()`. When clicked (returns `True`), constructs a hardcoded sample customer dict, calls `POST /predict` with the slider values injected, and displays the result with `st.metric()`.

**Main area**
- Title and timestamp.
- KPI row: `st.columns(4)` creates four equal columns. Each column uses `st.metric()` to display a headline number.
- `st.divider()` — horizontal rule.
- Two-column section (`st.columns(2)`): prediction histogram (Plotly) and risk pie chart (Plotly).
- Full-width time-series line chart: `df.resample("2h")` aggregates the demo data into 2-hour buckets. `fig.add_hline()` adds a dashed red threshold line at 0.7.
- Full-width drift bar chart: hardcoded PSI scores for `tenure` and `MonthlyCharges`. Bars above `drift_threshold` (0.15) are coloured red. `st.warning()` or `st.success()` appears below based on whether any feature exceeded the threshold.

---

## 10. Tests — `tests/test_features.py`

**Purpose:** Unit tests for the feature engineering functions. Run with `pytest tests/ -v`.

**What is tested:** `clean_raw_data()` and `build_preprocessing_pipeline()` from `src/features/build.py`.

---

### How pytest fixtures work

Functions decorated with `@pytest.fixture` are **dependency injection providers**. When a test function declares a parameter with the same name as a fixture, pytest automatically calls the fixture and passes its return value as the argument. Fixtures run fresh for each test by default (`scope="function"`).

#### `@pytest.fixture — sample_cfg`
Returns a minimal config dict — only the keys that `clean_raw_data()` and `build_preprocessing_pipeline()` actually read. Avoids depending on the real `config.yaml` file being present or correctly formatted.

#### `@pytest.fixture — sample_df`
Returns a 3-row DataFrame with:
- `customerID` — to be dropped.
- `TotalCharges` as strings (`"780"`, `"1920"`, `" "`) — the `" "` (space) is the real-world bug the test is guarding against.
- A mix of churn values to test binary encoding.

---

### Test functions

Each test function name starts with `test_` — pytest's discovery convention.

#### `test_clean_raw_data_removes_customer_id(sample_df, sample_cfg)`
Asserts that `customerID` is not present in `X` after cleaning. Guards against regressions in the `drop` logic.

#### `test_clean_raw_data_encodes_target(sample_df, sample_cfg)`
Asserts that `y` contains only `{0, 1}` and that `"Yes"` maps to `1` and `"No"` maps to `0`.

#### `test_clean_raw_data_handles_empty_total_charges(sample_df, sample_cfg)`
Asserts two things:
1. No nulls remain in `TotalCharges` after cleaning.
2. The new customer (row 2, who had `" "` as `TotalCharges`) gets imputed with their `MonthlyCharges` value (50.0).

This test caught a real bug: `df["TotalCharges"].fillna(..., inplace=True)` silently failed in pandas 2.0+ due to Copy-on-Write semantics. The fix was `df["TotalCharges"] = df["TotalCharges"].fillna(...)`.

#### `test_preprocessing_pipeline_output_shape(sample_df, sample_cfg)`
Fits and transforms the preprocessor and asserts the output has exactly `len(numeric) + len(categorical)` columns. Guards against the ColumnTransformer accidentally including or dropping columns.

#### `test_preprocessing_handles_unseen_categories(sample_df, sample_cfg)`
Fits the preprocessor on the sample data (which has `"Month-to-month"`, `"Two year"`, `"One year"` as contract types), then transforms a new row with `"Quarterly"` — a category never seen during training.

- Asserts no exception is raised (the `OrdinalEncoder` with `unknown_value=-1` handles this).
- Asserts the output still has 5 columns. This directly validates the `handle_unknown="use_encoded_value"` setting.

---

## 11. Data Flow Summary

```
config.yaml
    │
    │ (read by every script)
    │
    ├──► ingest.py
    │        │
    │        └──► data/raw/telco.csv
    │                    │
    │                    ▼
    ├──► build.py ◄──────┘
    │        │
    │        ├──► data/processed/features.csv  (training reference + monitoring)
    │        └──► models/pipeline.pkl          (fitted ColumnTransformer)
    │                    │
    │                    ▼
    ├──► train.py ◄──────┘
    │        │
    │        └──► models/churn_model.pkl       (fitted XGBClassifier)
    │                    │
    │           ┌────────┴─────────┐
    │           ▼                  ▼
    │       api/main.py       dashboard/app.py
    │       (loads both)      (reads config,
    │       (port 8000)        calls API over HTTP)
    │                          (port 8501)
    │
    └──► tests/test_features.py
             (imports src/features/build.py directly, no file I/O)
```

### Artifact dependency chain

| Artifact | Created by | Consumed by |
|---|---|---|
| `data/raw/telco.csv` | `ingest.py` | `build.py` |
| `data/processed/features.csv` | `build.py` | `train.py`, `dashboard/app.py` (drift reference) |
| `models/pipeline.pkl` | `build.py` | `api/main.py` → `predict.py` |
| `models/churn_model.pkl` | `train.py` | `api/main.py` → `predict.py` |

### Request lifecycle (single prediction)

```
Client sends POST /predict  { "tenure": 24, "MonthlyCharges": 85, ... }
         │
         ▼
  FastAPI parses body into CustomerFeatures (Pydantic validates types/constraints)
         │
         ▼
  predict_churn() calls predict(customer.model_dump(), model, pipeline, cfg)
         │
         ▼
  predict() wraps dict in pd.DataFrame([customer_data])
         │
         ▼
  pipeline.transform(df)  →  applies same StandardScaler + OrdinalEncoder as training
         │
         ▼
  model.predict_proba(X)[0, 1]  →  churn probability float
         │
         ▼
  Threshold check  →  risk label ("Low" / "Medium" / "High")
         │
         ▼
  feature_importances_ lookup  →  top 5 risk factor names
         │
         ▼
  FastAPI serialises PredictionResponse  →  JSON response to client
```
