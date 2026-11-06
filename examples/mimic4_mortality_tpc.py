"""TPC on MIMIC-IV in-hospital mortality: model comparison and hyperparameter sweep.

PURPOSE OF THIS FILE
    The example script the rubric requires (named
    examples/{dataset}_{task_name}_{model}.py). It runs every experiment:
    trains each configuration, tests it, and appends one row of scores per
    run to a CSV. Our final findings go in the RESULTS section below.

Paper: Rocheteau, Lio & Hyland, "Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit", CHIL 2021.
https://arxiv.org/abs/2007.09483

Experiments:
    1. Model comparison: TPC vs. PyHealth's LSTM, Transformer and TCN.
    2. Hyperparameter sweep: one TPC setting changed at a time (dropout,
       temp_channels, point_size, learning rate).

Usage (--root = the MIMIC-IV folder containing hosp/ and icu/):
    python examples/mimic4_mortality_tpc.py --root /path/to/mimiciv/2.2 --smoke
    python examples/mimic4_mortality_tpc.py --root /path/to/mimiciv/2.2 \
        --cache-dir /path/to/cache --configs tpc lstm --csv my.csv

Only aggregate counts and scores are printed, never patient rows.

RESULTS
    TODO: fill in the summary table once the full MIMIC-IV runs are done.

STATUS: shell. Structure and configs are set; function bodies are TODO.
"""

import argparse

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN, TCN, TPC, Transformer
from pyhealth.tasks import InHospitalMortalityMIMIC4
from pyhealth.trainer import Trainer

# Default TPC settings; every sweep config changes exactly one of these.
TPC_BASE = dict(
    num_layers=5,
    dilation="linear",
    kernel_size=4,
    temp_channels=8,
    point_size=16,
    dropout=0.3,
)

# name -> (model class, model kwargs). "lr" is passed to the optimizer.
CONFIGS = {
    # Experiment 1: model comparison
    "tpc": (TPC, dict(TPC_BASE)),
    "lstm": (RNN, dict(rnn_type="LSTM")),
    "transformer": (Transformer, dict()),
    "tcn": (TCN, dict()),
    # Experiment 2: hyperparameter sweep (values to finalize as a team)
    "dropout_0.0": (TPC, dict(TPC_BASE, dropout=0.0)),
    "dropout_0.5": (TPC, dict(TPC_BASE, dropout=0.5)),
    "temp_channels_4": (TPC, dict(TPC_BASE, temp_channels=4)),
    "temp_channels_16": (TPC, dict(TPC_BASE, temp_channels=16)),
    "point_size_8": (TPC, dict(TPC_BASE, point_size=8)),
    "point_size_32": (TPC, dict(TPC_BASE, point_size=32)),
    "lr_1e-4": (TPC, dict(TPC_BASE, lr=1e-4)),
    "lr_1e-2": (TPC, dict(TPC_BASE, lr=1e-2)),
}


def load_samples(root, cache_dir=None):
    """Loads MIMIC-IV and applies the in-hospital mortality task.

    Args:
        root: MIMIC-IV folder containing hosp/ and icu/.
        cache_dir: where PyHealth keeps its processed copy (optional).
    """
    # TODO: MIMIC4EHRDataset(root=root, tables=["labevents"],
    # cache_dir=cache_dir).set_task(InHospitalMortalityMIMIC4())
    raise NotImplementedError


def run_one(name, samples, seed, args):
    """Trains and tests one configuration; returns one results row (dict)."""
    # TODO: seed torch/numpy/random; split_by_patient 70/10/20; build the
    # model; Trainer(metrics=["roc_auc", "pr_auc"]).train(...); evaluate on
    # test; return config, seed, sizes, deaths, n_params, AUROC, AUPRC, time.
    raise NotImplementedError


def main():
    """Parses arguments and runs the chosen configs and seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="MIMIC-IV folder (hosp/, icu/)")
    parser.add_argument("--cache-dir", default=None, help="PyHealth cache folder")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 2 epochs")
    parser.add_argument("--csv", default="tpc_results.csv")
    args = parser.parse_args()
    # TODO: load samples once with load_samples(args.root, args.cache_dir),
    # loop configs x seeds, append rows to args.csv, and skip rows already in
    # the CSV so a crashed run can resume.
    raise NotImplementedError


if __name__ == "__main__":
    main()
