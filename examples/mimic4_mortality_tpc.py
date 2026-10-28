"""TPC on MIMIC-IV in-hospital mortality: baselines, ablations and sweep.

Paper: Rocheteau, Lio & Hyland, "Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit", CHIL 2021.
https://arxiv.org/abs/2007.09483

Trains the LSTM, Transformer and TCN baselines, the original TPC and its
ablations on several patient splits, and appends one CSV row per run. Runs
already in the CSV with status "ok" are skipped, so a crashed run resumes.

Usage (MIMIC4_ROOT is the folder containing hosp/ and icu/):
    export MIMIC4_ROOT=/path/to/mimic-iv-demo/2.2
    python examples/mimic4_mortality_tpc.py --smoke       # 1 split, 2 epochs
    python examples/mimic4_mortality_tpc.py               # 8 configs x 5 splits
    python examples/mimic4_mortality_tpc.py --ablations   # TPC ablations + sweep
    python examples/mimic4_mortality_tpc.py --summarize   # write summary table

For full MIMIC-IV, point MIMIC4_ROOT at the full data and pass ``--full``
(one split, training seeds 0-2, batch 64, early stopping).

Results go to ``tpc_output/`` (override with TPC_RESULTS_DIR). Only aggregate
counts are printed, never individual rows.
"""

import argparse
import csv
import json
import os
import random
import resource
import sys
import time
import traceback
from statistics import mean, stdev

import numpy as np
import torch

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN, TCN, TPC, Transformer
from pyhealth.tasks import InHospitalMortalityMIMIC4
from pyhealth.trainer import Trainer

# Folder containing hosp/ and icu/ (DEMO_ROOT is accepted as an older name).
MIMIC4_ROOT = os.environ.get("MIMIC4_ROOT", os.environ.get("DEMO_ROOT"))
CACHE_DIR = os.environ.get(
    "TPC_CACHE_DIR", os.path.expanduser("~/.cache/pyhealth_tpc")
)
RESULTS_DIR = os.environ.get("TPC_RESULTS_DIR", "tpc_output")

TPC_BASE = dict(
    num_layers=5,
    dilation="linear",
    kernel_size=4,
    temp_channels=8,
    point_size=16,
    dropout=0.3,
)

# name -> (model class, kwargs)
CONFIGS = {
    "lstm": (RNN, dict(rnn_type="LSTM")),
    "transformer": (Transformer, dict()),
    "tcn": (TCN, dict()),
    "tpc_original": (TPC, dict(TPC_BASE)),
    "tpc_temporal_only": (TPC, dict(TPC_BASE, use_pointwise=False)),
    "tpc_pointwise_only": (TPC, dict(TPC_BASE, use_temporal=False)),
    "tpc_shared_weights": (TPC, dict(TPC_BASE, share_weights=True)),
    "tpc_no_skips": (TPC, dict(TPC_BASE, use_skips=False)),
}
OPTIONAL_CONFIGS = {
    "tpc_exp_dilation": (TPC, dict(TPC_BASE, num_layers=4, dilation="exponential")),
}

# Extra ablations of the original TPC (one change each). "lr" is passed to the
# optimizer rather than the model.
ABLATION_CONFIGS = {
    "norm_batch": (TPC, dict(TPC_BASE, norm="batch")),
    "norm_none": (TPC, dict(TPC_BASE, norm="none")),
    "layers_3": (TPC, dict(TPC_BASE, num_layers=3)),
    "layers_7": (TPC, dict(TPC_BASE, num_layers=7)),
    "dropout_0.0": (TPC, dict(TPC_BASE, dropout=0.0)),
    "dropout_0.5": (TPC, dict(TPC_BASE, dropout=0.5)),
    "temp_channels_4": (TPC, dict(TPC_BASE, temp_channels=4)),
    "temp_channels_16": (TPC, dict(TPC_BASE, temp_channels=16)),
    "point_size_8": (TPC, dict(TPC_BASE, point_size=8)),
    "point_size_32": (TPC, dict(TPC_BASE, point_size=32)),
    "lr_1e-4": (TPC, dict(TPC_BASE, lr=1e-4)),
    "lr_1e-2": (TPC, dict(TPC_BASE, lr=1e-2)),
}

COLUMNS = [
    "config", "model", "split_seed", "n_train", "n_val", "n_test",
    "deaths_train", "deaths_test", "n_params", "receptive_field_hours",
    "epochs", "train_seconds", "test_roc_auc", "test_pr_auc", "test_loss",
    "status", "error", "peak_rss_mb",
]


def load_samples(root: str = MIMIC4_ROOT, cache_dir: str = CACHE_DIR):
    """Loads the dataset and applies the in-hospital mortality task."""
    if not root:
        sys.exit("Set MIMIC4_ROOT to the MIMIC-IV folder containing hosp/ and icu/.")
    base = MIMIC4EHRDataset(root=root, tables=["labevents"], cache_dir=cache_dir)
    return base.set_task(InHospitalMortalityMIMIC4(), num_workers=1)


def count_deaths(ds) -> int:
    return int(sum(int(s["mortality"]) for s in ds))


def make_split(samples, seed: int, max_redraws: int = 20):
    """70/10/20 patient split whose val and test sets each contain a death.

    Re-draws with seed + 100, seed + 200, ... when a split has none.
    Returns (train, val, test, seed_used).
    """
    for k in range(max_redraws):
        s = seed + 100 * k
        train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=s)
        if count_deaths(val) > 0 and count_deaths(test) > 0:
            return train, val, test, s
    raise RuntimeError(f"No split with deaths in val and test from seed {seed}.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def receptive_field(model) -> str:
    """Hours of history visible from the last step."""
    if isinstance(model, TPC):
        return str(model.receptive_field)
    if isinstance(model, TCN):
        rf = 1
        for m in model.modules():
            if isinstance(m, torch.nn.Conv1d) and m.kernel_size[0] > 1:
                rf += (m.kernel_size[0] - 1) * m.dilation[0]
        return str(rf)
    return "whole"


def read_rows(path: str):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def run_one(name, model_cls, kwargs, samples, split, seed, train_seed, args):
    train, val, test, split_seed = split
    row = dict(
        config=name, model=model_cls.__name__, split_seed=split_seed,
        n_train=len(train), n_val=len(val), n_test=len(test),
        deaths_train=count_deaths(train), deaths_test=count_deaths(test),
        epochs=args.epochs,
    )
    try:
        set_seed(train_seed)
        kwargs = dict(kwargs)
        lr = kwargs.pop("lr", 1e-3)
        model = model_cls(dataset=samples, **kwargs)
        row["n_params"] = sum(p.numel() for p in model.parameters())
        row["receptive_field_hours"] = receptive_field(model)
        train_dl = get_dataloader(train, batch_size=args.batch_size, shuffle=True)
        val_dl = get_dataloader(val, batch_size=args.batch_size, shuffle=False)
        test_dl = get_dataloader(test, batch_size=args.batch_size, shuffle=False)
        trainer = Trainer(
            model=model, metrics=["roc_auc", "pr_auc"],
            output_path=os.path.join(RESULTS_DIR, "logs"),
            exp_name=f"{name}_s{seed}",
        )
        t0 = time.time()
        train_kwargs = dict(
            train_dataloader=train_dl, val_dataloader=val_dl, epochs=args.epochs,
            optimizer_params={"lr": lr},
        )
        if args.full:
            train_kwargs.update(
                monitor="roc_auc", patience=5, load_best_model_at_last=True
            )
        else:
            train_kwargs.update(load_best_model_at_last=False)
        trainer.train(**train_kwargs)
        row["train_seconds"] = round(time.time() - t0, 2)
        scores = trainer.evaluate(test_dl)
        row["test_roc_auc"] = scores["roc_auc"]
        row["test_pr_auc"] = scores["pr_auc"]
        row["test_loss"] = scores["loss"]
        bad = [k for k in ("roc_auc", "pr_auc", "loss") if not np.isfinite(scores[k])]
        row["status"] = "failed" if bad else "ok"
        row["error"] = f"non-finite metrics: {bad}" if bad else ""
    except Exception as e:  # keep going; failures get a row
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
        traceback.print_exc()
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row["peak_rss_mb"] = round(rss / (2**20 if sys.platform == "darwin" else 2**10), 1)
    return row


def dataset_stats(samples) -> dict:
    hours = [int(s["labs"].shape[0]) for s in samples]
    labs = [s["labs"] for s in samples]
    zeros = sum(int((x == 0).sum()) for x in labs)
    total = sum(x.numel() for x in labs)
    deaths = count_deaths(samples)
    return dict(
        n_samples=len(samples),
        n_patients=len({s["patient_id"] for s in samples}),
        n_deaths=deaths,
        death_rate=deaths / max(len(samples), 1),
        n_features=int(labs[0].shape[1]) if labs else 0,
        hours_min=min(hours), hours_median=float(np.median(hours)),
        hours_max=max(hours),
        zero_share=zeros / max(total, 1),
    )


def fmt(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{mean(xs):.3f} ± {stdev(xs):.3f}"


def summarize(csv_path: str, out_path: str) -> None:
    rows = read_rows(csv_path)
    # The optional row 9 is kept in its own CSV so round1.csv stays 8 x 5.
    rows += read_rows(os.path.join(RESULTS_DIR, "round1_row9.csv"))
    stats_path = os.path.join(RESULTS_DIR, "round1_dataset.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    lines = ["# Round 1 summary", ""]
    if stats:
        lines += [
            "## Dataset (MIMIC-IV demo v2.2, InHospitalMortalityMIMIC4)", "",
            f"- Samples: {stats['n_samples']}, patients: {stats['n_patients']}, "
            f"deaths: {stats['n_deaths']}",
            f"- Death rate: {stats['death_rate']:.3f} (AUPRC of a random model)",
            f"- Lab features: {stats['n_features']}",
            f"- Hours per sample: min {stats['hours_min']}, median "
            f"{stats['hours_median']:g}, max {stats['hours_max']}",
            f"- Share of zero entries in the labs tensor: {stats['zero_share']:.3f}",
            "",
        ]
    lines += [
        "## Results (test set, mean ± std across splits)", "",
        "| Config | Runs ok | AUROC | AUPRC | Params | Receptive field (h) "
        "| Train s/run |",
        "|---|---|---|---|---|---|---|",
    ]
    order = list(CONFIGS) + list(OPTIONAL_CONFIGS) + list(ABLATION_CONFIGS)
    for name in order:
        rs = [r for r in rows if r["config"] == name]
        if not rs:
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        f = lambda k: [float(r[k]) for r in ok if r[k] != ""]
        params = ok[0]["n_params"] if ok else (rs[0]["n_params"] or "n/a")
        rf = rs[0]["receptive_field_hours"] or "n/a"
        secs = f"{mean(f('train_seconds')):.1f}" if ok else "n/a"
        if str(params).isdigit():
            params = f"{int(params):,}"
        lines.append(
            f"| {name} | {len(ok)}/{len(rs)} | {fmt(f('test_roc_auc'))} | "
            f"{fmt(f('test_pr_auc'))} | {params} | {rf} | {secs} |"
        )
    seeds = sorted({int(r["split_seed"]) for r in rows})
    lines += ["", f"Split seeds used: {seeds}", ""]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="1 split, 2 epochs")
    p.add_argument("--full", action="store_true", help="full MIMIC-IV settings")
    p.add_argument("--optional", action="store_true", help="also run row 9")
    p.add_argument("--ablations", action="store_true", help="run ABLATION_CONFIGS")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    if args.full:
        args.epochs, args.batch_size = 30, 64
    elif args.smoke:
        args.epochs, args.batch_size = 2, 32
    else:
        args.epochs, args.batch_size = 15, 32
    csv_path = args.csv or os.path.join(
        RESULTS_DIR,
        "round1_smoke.csv" if args.smoke
        else "round1_ablations.csv" if args.ablations else "round1.csv",
    )
    if args.summarize:
        summarize(csv_path, os.path.join(RESULTS_DIR, "round1_summary.md"))
        return

    configs = dict(CONFIGS, **(OPTIONAL_CONFIGS if args.optional else {}))
    if args.ablations:
        configs = dict(ABLATION_CONFIGS)
    if args.configs:
        all_configs = {**CONFIGS, **OPTIONAL_CONFIGS, **ABLATION_CONFIGS}
        configs = {k: v for k, v in all_configs.items()
                   if k in args.configs}

    samples = load_samples()
    stats = dataset_stats(samples)
    print("Dataset:", json.dumps(stats))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not args.smoke:
        with open(os.path.join(RESULTS_DIR, "round1_dataset.json"), "w") as f:
            json.dump(stats, f, indent=2)

    # (split seed, training seed) pairs.
    if args.full:
        plan = [(0, t) for t in range(3)]
    elif args.smoke:
        plan = [(0, 0)]
    else:
        plan = [(s, s) for s in range(5)]

    rows = read_rows(csv_path)
    done = {(r["config"], r["split_seed"], r.get("epochs")) for r in rows
            if r["status"] == "ok"}
    splits = {}
    for seed, train_seed in plan:
        if seed not in splits:
            splits[seed] = make_split(samples, seed)
            tr, va, te, used = splits[seed]
            note = "" if used == seed else f" (re-drawn from seed {seed})"
            print(f"Split seed {used}{note}: n={len(tr)}/{len(va)}/{len(te)}, "
                  f"deaths={count_deaths(tr)}/{count_deaths(va)}/{count_deaths(te)}")
        split = splits[seed]
        for name, (cls, kwargs) in configs.items():
            if (name, str(split[3]), str(args.epochs)) in done and not args.full:
                print(f"skip {name} split {split[3]} (already in CSV)")
                continue
            print(f"=== {name} | split {split[3]} | train seed {train_seed} ===")
            row = run_one(name, cls, kwargs, samples, split, seed, train_seed, args)
            rows = [r for r in rows if not (
                r["config"] == name and r["split_seed"] == str(split[3])
                and r["status"] != "ok")]
            rows.append(row)
            write_rows(csv_path, rows)
            print(f"-> {row['status']} auroc={row.get('test_roc_auc')} "
                  f"auprc={row.get('test_pr_auc')} {row.get('error', '')}")


if __name__ == "__main__":
    main()
