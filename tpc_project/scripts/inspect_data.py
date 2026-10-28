"""Step 4: aggregate statistics of the in-hospital mortality samples.

Prints counts and distributions only, never individual rows.
"""

import os
import sys

import numpy as np

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sys.path.insert(0, EXAMPLES)
from mimic4_mortality_tpc import dataset_stats, load_samples  # noqa: E402

from pyhealth.tasks import InHospitalMortalityMIMIC4  # noqa: E402


def main():
    samples = load_samples()
    stats = dataset_stats(samples)
    hours = np.array([int(s["labs"].shape[0]) for s in samples])
    print(f"Samples: {stats['n_samples']}")
    print(f"Patients: {stats['n_patients']}")
    print(f"Deaths: {stats['n_deaths']} (rate {stats['death_rate']:.3f})")
    print(f"Lab features: {stats['n_features']} "
          f"(task lists {len(InHospitalMortalityMIMIC4.LABITEMS)} item IDs)")
    print(f"Hours per sample: min {hours.min()}, median {np.median(hours):g}, "
          f"max {hours.max()}")
    counts, edges = np.histogram(hours, bins=[1, 2, 6, 12, 24, 36, 48, 50])
    bins = [f"{int(a)}-{int(b) - 1}h: {int(c)}" for a, b, c in zip(edges, edges[1:], counts)]
    print("Hours histogram:", ", ".join(bins))
    print(f"Share of zero entries: {stats['zero_share']:.3f}")
    # Zeros before a lab's first value (the processor's fill) vs. real zeros.
    leading = total = 0
    for s in samples:
        x = s["labs"].numpy()
        nz = x != 0
        first = np.where(nz.any(0), nz.argmax(0), x.shape[0])
        leading += int(first.sum())
        total += x.size
    print(f"Share of entries that are pre-first-measurement fill: {leading / total:.3f}")
    never = sum(int((s["labs"].numpy() != 0).any(0).sum() == 0) for s in samples)
    per_feature = np.mean([(s["labs"].numpy() != 0).any(0) for s in samples], axis=0)
    print(f"Samples with no labs at all: {never}")
    print(f"Features observed in >=1 hour, per-sample share (min/median/max): "
          f"{per_feature.min():.2f}/{np.median(per_feature):.2f}/{per_feature.max():.2f}")
    print(f"Features never observed in any sample: {int((per_feature == 0).sum())}")
    ok = stats["n_features"] == len(InHospitalMortalityMIMIC4.LABITEMS) and hours.max() <= 49
    print("CHECK", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
