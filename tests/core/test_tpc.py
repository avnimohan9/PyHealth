"""Unit tests for the TPC model.

PURPOSE OF THIS FILE
    Fast tests the rubric requires: instantiation, forward pass, output shapes
    and gradients. They must use small synthetic data (never MIMIC, not even
    the demo) and finish in milliseconds.

Run: python -m pytest tests/core/test_tpc.py -q

STATUS: implemented.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC
from pyhealth.models.tpc import MaskedNorm1d


def _make_patient(pid, hours, label, rng, n_features=5):
    """One synthetic patient: sparse random values on an hourly grid.

    Like the real task data, a missing lab is 0 and every hour has at least
    one measured lab.
    """
    t0 = datetime(2020, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(hours)]
    values = rng.normal(100, 20, size=(hours, n_features))
    values[rng.random(values.shape) < 0.6] = 0.0  # most labs unmeasured
    values[np.arange(hours), rng.integers(0, n_features, size=hours)] = 50.0  # >= 1 lab per hour
    return {"patient_id": pid, "labs": (times, values), "mortality": label}


class TestTPC(unittest.TestCase):
    """Test cases for the TPC model."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        # Two patients of different lengths, so the batch is padded.
        samples = [
            _make_patient("patient-0", 12, 1, rng),
            _make_patient("patient-1", 5, 0, rng),
        ]
        cls.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"labs": "timeseries"},
            output_schema={"mortality": "binary"},
            dataset_name="test_tpc",
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=2)))

    def test_model_initialization(self):
        """The model reads feature and label keys from the dataset."""
        model = TPC(dataset=self.dataset, num_layers=2)
        self.assertEqual(list(model.feature_keys), ["labs"])
        self.assertEqual(model.label_key, "mortality")
        self.assertEqual(model.dilations, [1, 2])
        self.assertEqual(model.num_features, 5)

    def test_model_forward(self):
        """Forward returns loss, y_prob, y_true and logit with right shapes."""
        model = TPC(dataset=self.dataset, num_layers=2)
        out = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out)
        self.assertEqual(tuple(out["logit"].shape), (2, 1))
        self.assertEqual(tuple(out["y_prob"].shape), (2, 1))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.all((out["y_prob"] >= 0) & (out["y_prob"] <= 1)))

    def test_model_backward(self):
        """Every trainable parameter receives a finite gradient."""
        model = TPC(dataset=self.dataset, num_layers=2)
        model(**self.batch)["loss"].backward()
        for name, param in model.named_parameters():
            if param.numel() == 0:  # BaseModel's placeholder parameter
                continue
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_ablation_flags(self):
        """Each option (use_temporal, use_pointwise, ...) runs fwd and bwd."""
        configs = [
            dict(use_temporal=False),
            dict(use_pointwise=False),
            dict(use_skips=False),
            dict(use_temporal=False, use_skips=False),
            dict(use_pointwise=False, use_skips=False),
            dict(share_weights=True),
            dict(dilation="exponential"),
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                model = TPC(dataset=self.dataset, num_layers=3, **cfg)
                out = model(**self.batch)
                self.assertEqual(tuple(out["logit"].shape), (2, 1))
                out["loss"].backward()
                self.assertTrue(torch.isfinite(out["loss"]))

    def test_invalid_arguments(self):
        """Bad configurations raise ValueError."""
        bad = [
            dict(use_temporal=False, use_pointwise=False),
            dict(num_layers=0),
            dict(dilation="bogus"),
            dict(dropout=1.0),
            dict(kernel_size=0),
        ]
        for cfg in bad:
            with self.subTest(**cfg):
                with self.assertRaises(ValueError):
                    TPC(dataset=self.dataset, **cfg)

    def test_receptive_field(self):
        """5 linear layers with kernel_size 4 see 46 hours."""
        model = TPC(dataset=self.dataset, num_layers=5, kernel_size=4)
        self.assertEqual(model.receptive_field, 46)
        self.assertEqual(TPC(dataset=self.dataset, num_layers=5, use_temporal=False).receptive_field, 1)
        exp = TPC(dataset=self.dataset, num_layers=3, kernel_size=3, dilation="exponential")
        self.assertEqual(exp.receptive_field, 1 + 2 * (1 + 2 + 4))

    def test_padding_does_not_change_prediction(self):
        """A padded patient gets the same output as when run alone (eval)."""
        model = TPC(dataset=self.dataset, num_layers=2)
        model.eval()
        with torch.no_grad():
            together = model(**self.batch)["logit"]
            alone = torch.cat(
                [model(**b)["logit"] for b in get_dataloader(self.dataset, batch_size=1)]
            )
        torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-4)


class TestMaskedNorm1d(unittest.TestCase):
    """Test cases for the padding-aware normalization."""

    def test_ignores_padding(self):
        """Statistics use real steps only and padded steps stay zero."""
        torch.manual_seed(0)
        norm = MaskedNorm1d(3)
        norm.train()
        x = torch.randn(2, 3, 6)
        mask = torch.tensor([[True] * 6, [True] * 2 + [False] * 4])
        x_garbage = x.clone()
        x_garbage[1, :, 2:] = 99.0  # junk in the padded steps

        out = norm(x_garbage, mask)

        self.assertTrue(torch.all(out[1, :, 2:] == 0))  # padded steps are zero
        real = torch.cat([out[0], out[1, :, :2]], dim=1)  # [3, 8] real steps
        torch.testing.assert_close(real.mean(dim=1), torch.zeros(3), atol=1e-5, rtol=0)
        torch.testing.assert_close(real.var(dim=1, unbiased=False), torch.ones(3), atol=1e-3, rtol=0)
        # The junk in padded steps did not influence the real outputs.
        torch.testing.assert_close(out, norm(x, mask))


if __name__ == "__main__":
    unittest.main()
