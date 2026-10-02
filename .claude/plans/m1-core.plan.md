# Plan — Milestone 1: Trustworthy tabular+text core

## Context
PRD `.claude/prds/no-code-ml-v2.prd.md`, milestone 1. A user points a local tool at a table or text dataset and gets:
- a **validity report**
- a **zero-training baseline**
- a trained model, **only if it beats the baseline**

Everything runs on their own machine, and several runs can happen at once without interfering.

Today `main` is a single-user Flask app (`main.py`, `nocode.py`, `mllib.py`) with:
- a process-global `nocode` object
- regression only
- a fixed 80/20 split
- no checks, no tests

The Copilot branch is a FastAPI + Create React App rewrite (CRA is deprecated). It is a web-SaaS shape, which the PRD has dropped.

**Baseline decision: start fresh and keep it small.** Copy only the null-placeholder normalisation idea from Copilot's `backend/data_processor/cleaner.py:clean_dataframe`. The rest of Copilot's code is reference only. Its big Optuna/model zoo is out: tuning is YAGNI until parity metrics show a gap.

## Approach
A local Python package plus CLI. Each run is a separate process writing to its own `runs/<id>/` directory, so there is no shared state. That covers concurrency without a server.

**Model tiers, all compared in one CV report:**

| Tier | What | Notes |
|---|---|---|
| 0 | Floor: `DummyClassifier` / `DummyRegressor` | Every score is shown relative to this. |
| 1 | Zero-training: **TabICLv2** (`tabicl`, BSD-3, commercial OK, sklearn API, runs on CPU) | Numeric and categorical columns. Optional extra `[fm]` because it pulls in torch. If it isn't installed, the report says "skipped, install `[fm]`". |
| 2 | Cheap training | Text columns: TF-IDF + `LogisticRegression` / `Ridge`. Tabular: sklearn `HistGradientBoosting*` with defaults, which handles NaNs natively. Seconds on CPU, no new deps. |

- **Selection rule:** tier 2 wins only if it beats tier 1 by more than one CV standard deviation. Otherwise the zero-training model is kept.
- **Pure-text zero-training** (embeddings or an LLM teacher) is milestone 2. In milestone 1, text gets tiers 0 and 2 only, and the report says so.

**Validity checks, which appear in the report before any score:**
- **Leakage/proxy:** any single feature whose shallow-tree CV score is at least 0.98 of perfect, or that maps 1:1 to the target. The column is flagged and excluded.
- **ID-like columns:** unique per row and not numeric-continuous. Flagged and excluded.
- **Contamination:** duplicate feature rows (all of them counted), and identical rows with conflicting targets.
- **Imbalance:** minority class under 5%, so the report uses balanced metrics.
- **Too small:** fewer than 50 rows, or fewer than 10 rows per class.
- **Missing target:** rows without a target are dropped and counted.
- **Datetime column present:** warns that random CV may overstate performance.

**Evaluation:** 5-fold CV, stratified for classification.
- Classification: balanced accuracy and macro-F1.
- Regression: MAE and R².

**Outputs** in `runs/<UTC timestamp>-<short uuid>/`:
- `report.json`
- `report.md` (warnings first, then a tier table)
- `model.joblib`, the winning full sklearn pipeline refit on all data

## Files (all new unless marked)
- `pyproject.toml`
  - deps: `pandas>=2.2`, `scikit-learn>=1.6`
  - extras: `fm = ["tabicl"]`, `dev = ["pytest"]`
  - script: `nocodeml = "nocodeml.cli:main"`
  - The package name is a placeholder; the product name is still open.
- `nocodeml/checks.py`: `run_checks(df, target) -> (warnings: list[dict], excluded_cols: list[str])`
- `nocodeml/core.py`:
  - `load(path)` reads CSV, XLSX or Parquet and normalises null placeholders.
  - `detect_task(y)`
  - `column_kinds(df)` sorts columns into numeric, categorical or text. Text means an object column with average length over 30 chars and high cardinality.
  - `build_candidates(kinds, task)` returns the tiers.
  - `evaluate(df, target) -> report`
  - `save(run_dir, report, model)`
- `nocodeml/cli.py` (argparse):
  - `nocodeml train DATA --target COL [--out runs/]`
  - `nocodeml predict RUN_DIR DATA [--out preds.csv]`
  - If `--target` is missing, it errors and lists the columns. Natural-language target inference is milestone 2.
- `tests/test_core.py`
- **Delete:** `main.py`, `nocode.py`, `mllib.py`, `requirements.txt` (UTF-16, 2023 pins), `static/`, `templates/`. They stay in git history.
- **Modify:** `README.md` with install, train and predict instructions (short). Also `.gitignore` to add `runs/` and `.venv/`.

## Steps (loop checklist → copy to `.claude/plans/m1-core.plan.md`)
- [ ] 0. Create branch `loop/m1-core`. Copy this plan into `.claude/plans/m1-core.plan.md`. Add `pyproject.toml` and an empty package. Get `python -m pytest -q` green with one smoke test.
- [ ] 1. `load` and `detect_task`, plus tests: a CSV round-trip, null placeholders becoming NaN, and classification vs regression detection.
- [ ] 2. `checks.py`, plus tests on synthetic data: a planted leak column is excluded, an ID column is excluded, imbalance is flagged, a tiny dataset is flagged, and conflicting duplicates are counted.
- [ ] 3. Tiers 0 and 2 plus CV evaluation and the selection rule. Tests: regression via `make_regression` and classification via `make_classification` both beat the dummy; a text dataset (inline synthetic sentences) works.
- [ ] 4. Tier 1 TabICL behind `[fm]`. Test with `pytest.importorskip("tabicl")`. Without the extra, the report shows "skipped".
- [ ] 5. `save` and the CLI `train`/`predict`. Tests: an end-to-end subprocess run writes the three artifacts, and `predict` returns one prediction per row.
- [ ] 6. Concurrency test: two `train` subprocesses on different datasets in parallel both succeed, with separate run dirs and correct reports.
- [ ] 7. Delete the old Flask app, update the README and `.gitignore`. Full suite green.

## Verification
- `python -m venv .venv && .venv/bin/pip install -e ".[dev,fm]" && .venv/bin/python -m pytest -q`
- Manual run on a public dataset with a planted leak:
  - `nocodeml train titanic.csv --target Survived`
  - Check that `report.md` lists the warnings, excludes `PassengerId` as ID-like, shows a tier table with the dummy floor, and names the winning tier with a reason.
  - `nocodeml predict runs/<id> titanic.csv`
- Two terminals running `train` at once both finish with distinct run dirs.

## Deliberately skipped (with ceiling)
| Skipped | Add it when |
|---|---|
| Hyperparameter tuning | Parity data shows a gap. |
| Notebook export | Milestone 3. |
| Web UI and server | A human checkpoint asks for it. |
| Embeddings or LLM teacher for text | Milestone 2. |
| Safer model format (skops instead of joblib) | Artifacts are shared between users. Today they only load locally, from your own runs. |
