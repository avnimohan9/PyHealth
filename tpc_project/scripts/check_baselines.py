"""Step 5: one batch through each PyHealth baseline on the real samples."""

import os
import sys

import torch

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
sys.path.insert(0, EXAMPLES)
from mimic4_mortality_tpc import CONFIGS, load_samples  # noqa: E402

from pyhealth.datasets import get_dataloader  # noqa: E402


def main():
    samples = load_samples()
    batch = next(iter(get_dataloader(samples, batch_size=32, shuffle=False)))
    print("labs batch shape:", tuple(batch["labs"].shape))
    for name in ("lstm", "transformer", "tcn"):
        cls, kwargs = CONFIGS[name]
        try:
            torch.manual_seed(0)
            model = cls(dataset=samples, **kwargs)
            out = model(**batch)
            out["loss"].backward()
            ok = "loss" in out and "y_prob" in out and torch.isfinite(out["loss"])
            print(f"{name:12s} {'PASS' if ok else 'FAIL'} loss={out['loss'].item():.4f} "
                  f"y_prob={tuple(out['y_prob'].shape)} "
                  f"params={sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"{name:12s} FAIL {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
