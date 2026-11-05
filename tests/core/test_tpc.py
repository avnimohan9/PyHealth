"""Unit tests for the TPC model.

PURPOSE OF THIS FILE
    Fast tests the rubric requires: instantiation, forward pass, output shapes
    and gradients. They must use small synthetic data (never MIMIC, not even
    the demo) and finish in milliseconds.

Run: python -m pytest tests/core/test_tpc.py -q

STATUS: shell. The synthetic dataset is ready; each test skips until written.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np

from pyhealth.datasets import create_sample_dataset, get_dataloader


def _make_patient(pid, hours, label, rng, n_features=5):
    """One synthetic patient: sparse random values on an hourly grid."""
    t0 = datetime(2020, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(hours)]
    values = rng.normal(100, 20, size=(hours, n_features))
    values[rng.random(values.shape) < 0.6] = np.nan  # most hours unmeasured
    values[0] = rng.normal(100, 20, size=n_features)  # first hour measured
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
        # TODO: build TPC(dataset=self.dataset, num_layers=2) and check
        # feature_keys == ["labs"], label_key == "mortality", dilations.
        self.skipTest("TODO")

    def test_model_forward(self):
        """Forward returns loss, y_prob, y_true and logit with right shapes."""
        # TODO: y_prob and logit shaped (2, 1); loss is a scalar; 0 <= y_prob <= 1.
        self.skipTest("TODO")

    def test_model_backward(self):
        """Every trainable parameter receives a finite gradient."""
        # TODO: loss.backward(); skip empty params (BaseModel's placeholder).
        self.skipTest("TODO")

    def test_ablation_flags(self):
        """Each option (use_temporal, use_pointwise, ...) runs fwd and bwd."""
        # TODO: loop over the flags with self.subTest(...).
        self.skipTest("TODO")

    def test_invalid_arguments(self):
        """Bad configurations raise ValueError."""
        # TODO: e.g. use_temporal=False and use_pointwise=False together.
        self.skipTest("TODO")

    def test_receptive_field(self):
        """5 linear layers with kernel_size 4 see 46 hours."""
        self.skipTest("TODO")

    def test_padding_does_not_change_prediction(self):
        """A padded patient gets the same output as when run alone (eval)."""
        self.skipTest("TODO")


class TestMaskedNorm1d(unittest.TestCase):
    """Test cases for the padding-aware normalization."""

    def test_ignores_padding(self):
        """Statistics use real steps only and padded steps stay zero."""
        self.skipTest("TODO")


if __name__ == "__main__":
    unittest.main()
