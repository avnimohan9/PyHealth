# Round 1 summary

## Dataset (MIMIC-IV demo v2.2, InHospitalMortalityMIMIC4)

- Samples: 216, patients: 99, deaths: 13
- Death rate: 0.060 (AUPRC of a random model)
- Lab features: 27
- Hours per sample: min 1, median 30.5, max 48
- Share of zero entries in the labs tensor: 0.724

## Results (test set, mean ± std across splits)

| Config | Runs ok | AUROC | AUPRC | Params | Receptive field (h) | Train s/run |
|---|---|---|---|---|---|---|
| lstm | 5/5 | 0.452 ± 0.135 | 0.106 ± 0.064 | 135,809 | whole | 3.6 |
| transformer | 5/5 | 0.429 ± 0.261 | 0.117 ± 0.084 | 201,473 | whole | 2.4 |
| tcn | 5/5 | 0.606 ± 0.220 | 0.267 ± 0.259 | 267,905 | 31 | 8.5 |
| tpc_original | 5/5 | 0.617 ± 0.242 | 0.223 ± 0.162 | 145,545 | 46 | 54.4 |
| tpc_temporal_only | 5/5 | 0.577 ± 0.186 | 0.160 ± 0.088 | 43,049 | 46 | 23.3 |
| tpc_pointwise_only | 5/5 | 0.409 ± 0.244 | 0.106 ± 0.066 | 8,449 | 1 | 1.3 |
| tpc_shared_weights | 5/5 | 0.628 ± 0.145 | 0.192 ± 0.161 | 66,361 | 46 | 51.8 |
| tpc_no_skips | 5/5 | 0.632 ± 0.224 | 0.177 ± 0.111 | 40,937 | 46 | 23.4 |
| tpc_exp_dilation | 5/5 | 0.685 ± 0.140 | 0.222 ± 0.153 | 102,209 | 46 | 40.0 |

Split seeds used: [0, 1, 3, 4, 102]

Row 9 (`tpc_exp_dilation`) is optional and lives in `round1_row9.csv`; `round1.csv` holds the 40 required runs. Split seed 2 had no death in val/test and was re-drawn as 102.

## What this shows

The pipeline works end to end: all 40 required runs and the 5 optional ones finished with finite metrics, with no code changes to PyHealth beyond the import line. The numbers do not rank the models. Every config's AUROC std (0.14–0.26) is about as large as the gap between the best and worst mean, test sets hold 1–4 deaths, and split 102 has a single test death, so its AUROC is the rank of one patient. Means happen to line up with the paper for full > temporal-only > pointwise-only (0.617 > 0.577 > 0.409) but not for shared vs. unshared weights (0.628 vs. 0.617); neither is evidence either way. Two things look suspicious rather than informative: LSTM and Transformer average below 0.5 AUROC, and the Transformer's test loss (0.74) is ~3x everyone else's. The likely cause is that baselines receive raw, unnormalized lab values (e.g. sodium ≈ 140) while TPC normalizes its input; this was not fixed, since the guide forbids changing baseline code. No config produced identical scores across splits or AUROC stuck at 0.5.
