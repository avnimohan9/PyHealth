# TPC in PyHealth (CS598 DL4H project)

We're adding the **Temporal Pointwise Convolution (TPC)** model to PyHealth and
testing it on in-hospital mortality prediction.

- **Paper:** Rocheteau, Liò & Hyland, *Temporal Pointwise Convolutional Networks
  for Length of Stay Prediction in the Intensive Care Unit*, CHIL 2021
  ([arXiv 2007.09483](https://arxiv.org/abs/2007.09483))
- **Authors' code:** [EmmaRocheteau/TPC-LoS-prediction](https://github.com/EmmaRocheteau/TPC-LoS-prediction) (MIT license)
- **Contribution type:** model only (rubric Option 2)
- **Branch:** `tpc-model`

## The idea in one minute

TPC reads a patient's hourly lab results and predicts whether they will die in
hospital. Each layer does two things side by side:

1. **Tracks each lab over time** (temporal convolution), for example sodium
   rising over 12 hours. Each lab gets its own trend detectors.
2. **Compares all labs together at each hour** (pointwise convolution), for
   example low sodium and high potassium at the same time.

Layers are stacked, and each one passes along the original values plus what
earlier layers found (skip connections). Five layers let the model look back
about 46 hours. The prediction is made from the patient's last real hour.

## What's in this branch

**Files that will go into the final PyHealth PR:**

| File | What it is |
|---|---|
| `pyhealth/models/tpc.py` | The TPC model (`TPC`, `TPCLayer`, `MaskedNorm1d`) |
| `pyhealth/models/__init__.py` | One added line: `from .tpc import TPC, TPCLayer` |
| `tests/core/test_tpc.py` | Unit tests on synthetic data (fast, no real data) |
| `examples/mimic4_mortality_tpc.py` | Runs baselines, ablations and the hyperparameter sweep |

**Project-only files (this folder; delete before opening the PR):**

| File | What it is |
|---|---|
| `tpc_project/README.md` | This guide |
| `tpc_project/scripts/inspect_data.py` | Prints dataset counts (patients, deaths, hours) |
| `tpc_project/scripts/check_baselines.py` | Checks that LSTM, Transformer and TCN run on this data |
| `tpc_project/results/` | Results from our first run on the MIMIC-IV demo |

## Setup

PyHealth needs **Python 3.12 or newer**.

```bash
git clone https://github.com/avnimohan9/PyHealth.git
cd PyHealth
git checkout tpc-model
conda create -n tpc python=3.12 -y
conda activate tpc
pip install -e . pytest
```

Check that everything works:

```bash
python -m pytest tests/core/test_tpc.py -q
```

## Getting the data

**MIMIC-IV demo** (100 patients, open access, no credentialing). Download the
five files we need into a folder **outside the repo**:

```bash
mkdir -p ~/mimic-iv-demo/2.2/hosp ~/mimic-iv-demo/2.2/icu
cd ~/mimic-iv-demo/2.2
B=https://physionet.org/files/mimic-iv-demo/2.2
for f in hosp/patients hosp/admissions hosp/labevents hosp/d_labitems icu/icustays; do
  curl -sSfL -o $f.csv.gz $B/$f.csv.gz
done
```

**Full MIMIC-IV** needs PhysioNet credentialing (CITI training + data use
agreement). Same folder layout; just point `MIMIC4_ROOT` at it.

> Never commit data, and never paste or upload raw MIMIC-IV rows to an AI tool
> or other online service. The scripts only print aggregate counts.

## Running experiments

Run from the repo root. Results are written to `tpc_output/`, which git ignores.

```bash
export MIMIC4_ROOT=~/mimic-iv-demo/2.2

python examples/mimic4_mortality_tpc.py --smoke        # quick check: 8 models, 2 epochs (~1 min)
python examples/mimic4_mortality_tpc.py                # baselines + paper ablations, 5 splits (~15 min on a laptop)
python examples/mimic4_mortality_tpc.py --ablations    # our ablations + hyperparameter sweep (~60 min)
python examples/mimic4_mortality_tpc.py --summarize    # write the summary table
python examples/mimic4_mortality_tpc.py --configs tpc_original lstm   # run only some models
```

For full MIMIC-IV, add `--full` (one split, 3 training seeds, batch 64, early
stopping). Expect it to need a GPU.

The data helper scripts:

```bash
python tpc_project/scripts/inspect_data.py
python tpc_project/scripts/check_baselines.py
```

## How we differ from the paper

| | Paper | Us |
|---|---|---|
| Question | Remaining ICU length of stay, predicted every hour (mortality is a side task) | In-hospital death, one prediction |
| Inputs | ~100: vitals, labs, age/sex/weight, diagnoses, "time since measured" signals | 27 lab tests (10 lab types), hourly |
| Task | Custom SQL pipeline: ICU stays > 5 h | PyHealth's existing `InHospitalMortalityMIMIC4`: hospital stays > 48 h, first 48 h of labs |
| Data | Full MIMIC-IV and eICU | MIMIC-IV demo so far; full MIMIC-IV once access arrives |

**What we changed in the authors' model code:**

- Rewrote it as a PyHealth `BaseModel`, so it works with PyHealth's data and trainer
- Kept the core TPC layer logic unchanged
- Removed inputs our task doesn't have (diagnoses, static features, decay indicators)
- One mortality prediction from the last real hour, instead of hourly predictions
- Normalization ignores the zero padding rows added to equalize patient lengths
- Dilation grows 1, 2, 3, 4, 5 (46 h reach); the authors' code uses 1, 3, 6, 9, 12
- Added options for ablations: `dilation="exponential"` and `norm="masked" | "batch" | "none"`

## Model options

```python
from pyhealth.models import TPC

model = TPC(
    dataset=samples,
    num_layers=5,             # depth; 5 layers ~ 46 h of history
    dilation="linear",        # or "exponential"
    use_temporal=True,        # False = pointwise-only (paper ablation)
    use_pointwise=True,       # False = temporal-only (paper ablation)
    share_weights=False,      # True = one trend detector for all labs (paper ablation)
    use_skips=True,           # False = no skip connections (paper ablation)
    temp_channels=8,          # trend patterns per lab
    point_size=16,            # cross-lab patterns per layer
    kernel_size=4,
    dropout=0.3,
    norm="masked",            # "batch" or "none" for the normalization ablation
)
```

## Results so far (MIMIC-IV demo)

These are a **pipeline check, not real findings**. The demo has 216 hospital
stays and only 13 deaths, so each test set holds 1–4 deaths and scores swing a
lot between splits. AUROC: 0.5 = coin flip, 1.0 = perfect.

| Model | AUROC (mean of 5 splits) |
|---|---|
| TPC (original) | 0.62 |
| TCN | 0.61 |
| LSTM | 0.45 |
| Transformer | 0.43 |

Hyperparameter sweep (one change at a time from the original TPC):

| Change | AUROC |
|---|---|
| dropout 0.0 / 0.5 | 0.59 / 0.65 |
| temp_channels 4 / 16 | 0.67 / 0.66 |
| point_size 8 / 32 | 0.66 / 0.62 |
| learning rate 0.0001 / 0.01 | 0.52 / 0.60 |

Full tables: `results/round1_summary.md` and `results/round1_ablations.md`.
Everything we did differently from the original setup guide, and problems we
hit: `results/round1_notes.md`.

**Known issues:**

- LSTM and Transformer get raw, unscaled lab values while TPC scales its own
  input, so they're at a disadvantage.
- 72% of the input is zeros filled in before a lab's first measurement.
- 12 of the 27 lab codes never appear in the demo.

## Still to do for the final PR

- [ ] Docs page: `docs/api/models/pyhealth.models.TPC.rst` and an entry in `docs/api/models.rst`
- [ ] Run on full MIMIC-IV (or a subset) once credentials arrive
- [ ] Decide the final hyperparameter sweep values for full data
- [ ] Add a `train_seed` column to the results CSV so `--full` runs can be told apart
- [ ] Delete `tpc_project/`, rebase on upstream `main`, open the PR with "Allow edits by maintainers"
