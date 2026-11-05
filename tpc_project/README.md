# TPC in PyHealth: team guide (branch `tpc-dev`)

We're adding the **Temporal Pointwise Convolution (TPC)** model to PyHealth and testing it on
in-hospital mortality with full MIMIC-IV. This branch starts from the latest PyHealth with **shell
files**: each has its purpose at the top, the structure in place, and `TODO`s to fill in.

- **Paper:** Rocheteau, Liò & Hyland, CHIL 2021 ([arXiv 2007.09483](https://arxiv.org/abs/2007.09483))
- **Authors' code (reference, MIT):** [EmmaRocheteau/TPC-LoS-prediction](https://github.com/EmmaRocheteau/TPC-LoS-prediction), see `models/tpc_model.py`
- **Contribution type:** model only (rubric Option 2)
- **Final PR:** `avnimohan9/PyHealth` → `sunlabuiuc/PyHealth` **`master`**, due Nov 29

## The files

**Go into the final PR (6 files):**

| File | Purpose | Status |
|---|---|---|
| `pyhealth/models/tpc.py` | The TPC model: `MaskedNorm1d`, `TPCLayer`, `TPC` | Shell |
| `pyhealth/models/__init__.py` | One line so `from pyhealth.models import TPC` works | Done |
| `tests/core/test_tpc.py` | Fast unit tests on fake patients (rubric requirement) | Shell (tests skip) |
| `examples/mimic4_mortality_tpc.py` | Runs all experiments; final results go in its top comment | Shell (configs set) |
| `docs/api/models/pyhealth.models.TPC.rst` | The model's docs page | Done |
| `docs/api/models.rst` | One line adding the page to the docs index | Done |

**Team-only (`tpc_project/`, deleted before the PR):**

| File | Purpose |
|---|---|
| `README.md` | This guide |
| `notebooks/00_setup_and_tests.ipynb` | Get the code into Colab, install, run tests, check data; optional push of code edits |
| `notebooks/01_data_check.ipynb` | Load MIMIC-IV and print aggregate counts; optional smaller `labevents` copy |
| `notebooks/02_experiments.ipynb` | Train and test on GPU; each person runs their share of configs |
| `notebooks/03_results.ipynb` | Combine everyone's CSVs into the results tables |

Open a notebook in Colab: **File → Open notebook → GitHub**, paste
`https://github.com/avnimohan9/PyHealth/tree/tpc-dev`, pick the notebook. Then **File → Save a copy in
Drive** so your edits are kept.

## What lives where

| Place | Holds | Notes |
|---|---|---|
| **GitHub** (`tpc-dev`) | All code | The only copy of the code. Notebooks clone it. |
| **Google Colab** | Running things | Tests, data checks, experiments on GPU, results tables |
| **Shared Drive folder** | MIMIC-IV data, result CSVs | Shared with our 3 credentialed accounts only. Never on GitHub. |

Suggested Drive layout (folder name is set at the top of each notebook):

```
DL4H_TPC/
  mimiciv/2.2/hosp/   patients, admissions, labevents, d_labitems
  mimiciv/2.2/icu/    icustays
  results/            results_<name>.csv, one per person
```

## Workflow

**Writing code** (`tpc.py`, the tests, the example script):

- Edit on GitHub (pencil icon), in VS Code (Live Share lets us type together live), or in Colab's file
  browser and push with the last section of `00_setup_and_tests`.
- One person per file at a time. Say in the group chat what you're working on.
- Run the tests before pushing: `python -m pytest tests/core/test_tpc.py -q`.
- Bigger changes: make a small branch off `tpc-dev` and open a PR into `tpc-dev` inside our fork, so
  someone can look first.

**Running experiments** (Colab, `02_experiments`):

1. Agree who runs which configs (table at the top of the notebook).
2. Each person runs the smoke test, then their full runs, writing their **own** CSV.
3. `03_results` combines the CSVs into the tables.

**Data rules:** never print or save raw MIMIC-IV rows (aggregates only), don't use Colab's AI assistant
on cells that touch the data, never commit data to GitHub.

## Suggested order

1. **Model** (`tpc.py`): `MaskedNorm1d` → `TPCLayer` → `TPC`. Use the authors' `temp_pointwise()` as
   the reference for one layer.
2. **Tests** (`test_tpc.py`): replace each `skipTest("TODO")` as the model takes shape.
3. **Example script**: `load_samples`, `run_one`, `main`; check with `--smoke` on the data.
4. **Experiments** on full MIMIC-IV in Colab, split across the three of us.
5. **Results** into the example's RESULTS section; slides for the video.
6. **PR**: make a clean branch from `tpc-dev`, delete `tpc_project/`, rebase on upstream `master`, run
   the tests, open the PR with "Allow edits by maintainers" ticked.

## Setup on your own computer (optional)

```bash
git clone -b tpc-dev https://github.com/avnimohan9/PyHealth.git
cd PyHealth
conda create -n tpc python=3.12 -y && conda activate tpc
pip install -e . pytest
python -m pytest tests/core/test_tpc.py -q
```
