# Round 1 notes

Run date: 2026-10-20. Hardware: MacBook CPU, conda env `tpc` (Python 3.12.15, torch 2.14.1).
Data: MIMIC-IV Clinical Database Demo v2.2 (open access, downloaded locally; not in this repo).

## Step checks

| Step | Check | Result |
|---|---|---|
| 1 Repo | `pytest tests/core/test_tcn.py` | PASS (7 passed) |
| 2 Data | 5 files present under the data root | PASS (gzip-valid; 100 patients, 107,727 lab rows) |
| 3 Model | synthetic fwd/bwd for every flag combination | PASS (6 TPC configs) after fixing the check itself, see below |
| 4 Inspect | 27 features, hours ≤ 49 | PASS (27 features, 1–48 h) |
| 5 Baselines | RNN/Transformer/TCN return `loss` and `y_prob` | PASS, all three accept timeseries as is |
| 6 Runner | `--smoke` finishes 8 configs | PASS (35 s total) |
| 7 Round 1 | 40 rows, no NaN | PASS (40/40 ok, 0 NaN; 14.4 min total, peak RSS 0.85 GB) |
| Row 9 | optional exponential dilation | PASS (5/5 ok, in `round1_row9.csv`) |

## Dataset counts

216 samples (admissions), 99 patients, 13 deaths (6.0%). Labs tensor: 27 features. Hours per sample: min 1, median 30.5, max 48; 20 samples have a single hour, 100 have 24–35 h, 67 have 36–47 h.
72.4% of tensor entries are zero, and all of those are the processor's fill before a lab's first measurement. 12 of the 27 item IDs never appear in the demo, so only 15 features carry any signal.

Splits (patient-level 70/10/20; deaths train/val/test):

| Split seed | n train/val/test | deaths |
|---|---|---|
| 0 | 143/22/51 | 7/2/4 |
| 1 | 160/23/33 | 8/2/3 |
| 102 (re-drawn from 2) | 143/21/52 | 9/3/1 |
| 3 | 162/17/37 | 9/1/3 |
| 4 | 149/33/34 | 10/1/2 |

## Deviations from the guide

1. **No starter `tpc.py`.** It was not on the machine. As you asked, `pyhealth/models/tpc.py` was ported from the authors' repo (`EmmaRocheteau/TPC-LoS-prediction`, `models/tpc_model.py`) with the constructor flags the guide lists and the Section 7 fixes already in (masked normalization, `num_layers=5` default).
2. **Repo** is branch `tpc-model` of a PyHealth fork, based on upstream commit 7af8779. These results were copied into `tpc_project/results/`; training logs are not included.
3. **Python 3.12 conda env `tpc`.** Upstream PyHealth now requires Python ≥ 3.12.
4. **Data downloaded file by file** from `physionet.org/files/mimic-iv-demo/2.2/`. The full ZIP was downloading at ~0.6 MB/min, so it was stopped. Only the 5 required files are present.
5. **Linear dilation is d_i = i + 1 (1..5)**, as the guide specifies, which gives 46 h. The authors' code uses d_0 = 1, d_i = i·(k−1) (1,3,6,9,12), which gives 94 h, wider than the 48 h window.
6. **Port-specific choices** (the guide does not cover these):
   - No decay-indicator mask channels, diagnoses or flat features: the task does not provide them.
   - Input features go through a masked normalization (no affine) before the first layer. Every BatchNorm in the layers is a masked per-channel norm with running stats; padded hours are set back to 0.
   - Prediction head is Linear(·, 32) → ReLU → Dropout → Linear(·, 1) on the last real hour.
   - `share_weights` is applied to the full TPC model. The authors only implement it in their temporal-only model.
   - `use_skips=False` follows the authors' current code: the temporal and pointwise branches run separately and are concatenated only at the end. Their commented-out older version let the pointwise branch see the previous temporal output; this port does not.
   - Pointwise-only feeds each layer the raw features plus all earlier pointwise outputs, as the authors' code does.
7. **Split seed 2 → 102.** Seed 2 had no death in val or test.
8. **Extra CSV column `peak_rss_mb`** (process peak memory after each run), added for the scale-up question. Row 9 is in its own CSV so `round1.csv` keeps exactly 40 rows.
9. **Training seed = split seed** for each run (`torch`, `numpy` and `random` seeded before building the model; the dataloader shuffle seed is drawn from torch after that).

## Problems and workarounds

- The first synthetic check failed: it asserted that every parameter had a gradient, and `BaseModel._dummy_param` (an empty device-tracking tensor) never gets one. The check now skips empty parameters; the model was not changed.
- No other failures.

## Surprising or suspicious

- **Baselines receive raw lab values.** `EmbeddingModel` passes timeseries through a linear layer with no normalization. Initial losses were high (Transformer 4.49, TCN 3.16 on one batch), and LSTM and Transformer average below 0.5 AUROC. The Transformer's test loss is 0.74 versus ~0.2 for every other model, so its probabilities are badly calibrated. TPC normalizes its input, so the comparison is not apples to apples. This was not fixed because the guide forbids changing baseline code. Options for round 2: a normalizing input processor in the task pipeline, or a thin wrapper.
- **Zero-fill dominates the input** (72% of entries). After normalization those zeros become large negative values that the models cannot tell apart from real low values. Not fixed (guide: limitation only).
- **Test sets have 1–4 deaths.** Split 102 has one test death, so its AUROC is the rank of one patient; for example, pointwise-only scored 0.00 there.
- **TPC is the slowest model** (~54 s/run for the full model versus ~3 s for the LSTM), because the channel count grows with every layer: (F + Zt)·(1 + k) channels feed the grouped convolution.
- Paper ordering: the means agree for full > temporal-only > pointwise-only (0.617 > 0.577 > 0.409) and disagree for shared versus unshared weights (0.628 vs 0.617). The std is 0.14–0.26, so neither result means anything yet.

## Feasibility verdicts (Section 8)

| Idea | Verdict | Evidence |
|---|---|---|
| TPC as a PyHealth model | **Go** | Steps 3–7 pass with `BaseModel`, `Trainer` and `InHospitalMortalityMIMIC4` unchanged; the only edit to an existing file is the import line. |
| Reproduce paper ablations | **Go** | Rows 4–8 finish 25/25 runs with finite metrics. Needs a decision on the no-skip variant (authors' current versus older code). |
| New ablation: exponential dilation | **Go** | Row 9 finishes 5/5. Matched 46 h reach with 4 layers and 102,209 params versus 145,545 for the original, so the comparison mixes dilation pattern with model size. Report both. |
| Baselines (LSTM, Transformer, TCN) | **Go with changes** | All accept the timeseries input as is, but on raw values; add input normalization outside the model code before comparing. |
| Hourly LoS task (Model + Task route) | **Go with changes** | The labs-only grid starts at the first lab draw, not admission. The median sample covers 30.5 of 48 h, 20/216 samples have 1 h, and 72% of entries are pre-measurement fill. Per-hour labels need a grid anchored at ICU admission plus vitals from `chartevents`. |
| Scale to full MIMIC-IV | **Go with changes** | Demo: 0.85 GB peak RSS, ~54 s per 15-epoch TPC run on 143 training samples. Full `labevents` is ~1,100× the demo's rows, so use the filtered 27-itemid copy. Training scales roughly linearly with samples, so expect hours per TPC run on CPU; plan on a GPU. |

## Files

- `pyhealth/models/tpc.py`: model (`TPC`, `TPCLayer`, `MaskedNorm1d`)
- `pyhealth/models/__init__.py`: one import line
- `examples/mimic4_mortality_tpc.py`: runner (`--smoke`, `--optional`, `--configs`, `--summarize`, `--full`)
- `tests/core/test_tpc.py`: unit tests (replaced the step 3 synthetic check)
- `tpc_project/scripts/{inspect_data,check_baselines}.py`: steps 4–5
- `tpc_project/results/`: `round1.csv`, `round1_row9.csv`, `round1_ablations.csv`, `round1_dataset.json`, `round1_summary.md`, `round1_ablations.md`
