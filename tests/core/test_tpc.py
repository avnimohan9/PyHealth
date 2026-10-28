import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC
from pyhealth.models.tpc import MaskedNorm1d


def _make_patient(pid, hours, label, rng, n_features=5):
    """Synthetic timeseries sample: sparse values on an hourly grid."""
    t0 = datetime(2020, 1, 1)
    times = [t0 + timedelta(hours=h) for h in range(hours)]
    values = rng.normal(100, 20, size=(hours, n_features))
    values[rng.random(values.shape) < 0.6] = np.nan
    values[0] = rng.normal(100, 20, size=n_features)
    return {"patient_id": pid, "labs": (times, values), "mortality": label}


class TestTPC(unittest.TestCase):
    """Test cases for the TPC model."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        # Different lengths so the batch is padded.
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

    def _model(self, **kwargs):
        torch.manual_seed(0)
        return TPC(dataset=self.dataset, num_layers=2, **kwargs)

    def test_model_initialization(self):
        """The model reads feature and label keys from the dataset."""
        model = self._model()
        self.assertIsInstance(model, TPC)
        self.assertEqual(model.feature_keys, ["labs"])
        self.assertEqual(model.label_key, "mortality")
        self.assertEqual(model.dilations, [1, 2])

    def test_model_forward(self):
        """Forward returns loss, probabilities, labels and logits."""
        model = self._model()
        with torch.no_grad():
            ret = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, ret)
        self.assertEqual(ret["y_prob"].shape, (2, 1))
        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(((ret["y_prob"] >= 0) & (ret["y_prob"] <= 1)).all())

    def test_model_backward(self):
        """Every trainable parameter receives a finite gradient."""
        model = self._model()
        model(**self.batch)["loss"].backward()
        for name, p in model.named_parameters():
            if p.numel() == 0:  # BaseModel's device placeholder
                continue
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_ablation_flags(self):
        """Each paper ablation and option runs forward and backward."""
        variants = [
            dict(use_pointwise=False),
            dict(use_temporal=False),
            dict(share_weights=True),
            dict(use_skips=False),
            dict(dilation="exponential"),
            dict(norm="batch"),
            dict(norm="none"),
        ]
        for kwargs in variants:
            with self.subTest(**kwargs):
                model = self._model(**kwargs)
                ret = model(**self.batch)
                ret["loss"].backward()
                self.assertEqual(ret["y_prob"].shape, (2, 1))

    def test_invalid_arguments(self):
        """Bad configurations raise clear errors."""
        with self.assertRaises(ValueError):
            self._model(use_temporal=False, use_pointwise=False)
        with self.assertRaises(ValueError):
            self._model(dilation="quadratic")
        with self.assertRaises(ValueError):
            self._model(norm="layer")

    def test_receptive_field(self):
        """Receptive field follows 1 + (k - 1) * sum(dilations)."""
        ds = self.dataset
        self.assertEqual(TPC(dataset=ds, num_layers=5).receptive_field, 46)
        self.assertEqual(
            TPC(dataset=ds, num_layers=4, dilation="exponential").receptive_field, 46
        )
        self.assertEqual(TPC(dataset=ds, use_temporal=False).receptive_field, 1)

    def test_embed(self):
        """embed=True returns one embedding per patient."""
        model = self._model()
        with torch.no_grad():
            ret = model(**self.batch, embed=True)
        self.assertEqual(ret["embed"].shape[0], 2)

    def test_padding_does_not_change_prediction(self):
        """A patient's output is the same alone or padded in a batch."""
        model = self._model()
        model.eval()
        with torch.no_grad():
            batched = model(**self.batch)["logit"]
            # Patient 1 (5 hours) is padded to 12 hours in the batch.
            single = {
                "labs": self.batch["labs"][1:2, :5],
                "mortality": self.batch["mortality"][1:2],
            }
            alone = model(**single)["logit"]
        self.assertTrue(torch.allclose(batched[1], alone[0], atol=1e-5))


class TestMaskedNorm1d(unittest.TestCase):
    """Test cases for the padding-aware normalization."""

    def test_ignores_padding(self):
        """Statistics use valid steps only and padded steps stay zero."""
        norm = MaskedNorm1d(1, affine=False)
        x = torch.tensor([[[140.0, 142.0, 0.0, 0.0]]])
        mask = torch.tensor([[True, True, False, False]])
        out = norm(x, mask)
        self.assertAlmostEqual(out[0, 0, :2].mean().item(), 0.0, places=5)
        self.assertTrue(torch.equal(out[0, 0, 2:], torch.zeros(2)))

    def test_eval_uses_running_stats(self):
        """In eval mode the running statistics are used."""
        norm = MaskedNorm1d(1, affine=False, momentum=1.0)
        x = torch.tensor([[[1.0, 3.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        norm(x, mask)
        norm.eval()
        self.assertAlmostEqual(norm.running_mean.item(), 2.0, places=5)
        out = norm(torch.tensor([[[2.0, 2.0]]]), mask)
        self.assertTrue(torch.allclose(out, torch.zeros_like(out), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
