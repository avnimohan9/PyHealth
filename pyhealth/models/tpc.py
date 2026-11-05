"""Temporal Pointwise Convolutional Networks (TPC).

PURPOSE OF THIS FILE
    The model itself. This is the main file of our PyHealth contribution.

Paper: Rocheteau, Lio & Hyland. "Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit." CHIL 2021.
https://arxiv.org/abs/2007.09483

Reference code (MIT license): https://github.com/EmmaRocheteau/TPC-LoS-prediction
(see ``models/tpc_model.py``, especially ``temp_pointwise()``).

The model reads hourly lab values, shaped [batch, hours, labs], and predicts
in-hospital death. Each TPC layer has two branches that run side by side:

    - Temporal convolution: tracks each lab over time, with its own weights per
      lab (a grouped, causal, dilated 1D convolution).
    - Pointwise convolution: combines all labs at each hour (a 1x1 convolution,
      i.e. a linear layer applied at every hour).

The branch outputs, plus the layer's inputs (skip connections), are
concatenated and passed to the next layer. The prediction is made from each
patient's last real (non-padded) hour.

STATUS: shell. Classes and method signatures are fixed; bodies are TODO.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class MaskedNorm1d(nn.Module):
    """Normalizes each channel using real (non-padded) time steps only.

    Why: batches are padded with zero rows so patients have equal length.
    Standard BatchNorm would count those rows when computing the mean and
    variance. This version ignores them and sets padded steps back to zero.

    Args:
        num_features: number of channels C.
        momentum: running-statistics momentum.
        affine: whether to learn a per-channel scale and shift.
        eps: numerical stability constant.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        # TODO: register running_mean / running_var buffers and, if affine,
        # weight / bias parameters.
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalizes x.

        Args:
            x: tensor of shape [batch, channels, hours].
            mask: bool tensor of shape [batch, hours]; True = real hour.

        Returns:
            Tensor of the same shape, with padded hours set to 0.
        """
        # TODO: training -> mean/var over real hours, update running stats;
        # eval -> use running stats. Multiply the output by the mask.
        raise NotImplementedError


class TPCLayer(nn.Module):
    """One TPC layer: temporal convolution + pointwise convolution + skips.

    Args:
        num_features: F, number of raw input features (labs).
        temp_channels: temporal output channels per feature.
        point_size: pointwise output size.
        kernel_size: hours compared by each temporal convolution.
        dilation: gap between those hours.
        dropout: dropout rate.
        share_weights: share one temporal kernel across all features.
        use_temporal: enable the temporal branch.
        use_pointwise: enable the pointwise branch.
        use_skips: concatenate inputs and earlier outputs (full model).
    """

    def __init__(
        self,
        num_features: int,
        temp_channels: int,
        point_size: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        share_weights: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skips: bool = True,
    ):
        super().__init__()
        # TODO: build the grouped Conv1d (temporal), the Linear (pointwise),
        # their normalization layers and dropout. Track the channel counts
        # that grow from layer to layer (see the authors' code, F + Zt).
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Runs one layer.

        Args:
            x: layer input of shape [batch, channels, hours].
            mask: bool tensor of shape [batch, hours]; True = real hour.

        Returns:
            Input for the next layer, shape [batch, channels_out, hours].
        """
        # TODO: left-pad by (kernel_size - 1) * dilation so the convolution
        # is causal, run both branches, concatenate with the skip inputs,
        # apply ReLU.
        raise NotImplementedError


class TPC(BaseModel):
    """Temporal Pointwise Convolutional Network for timeseries inputs.

    Each feature key must be a ``timeseries`` input of shape
    [batch, hours, features], for example the ``labs`` key of
    ``InHospitalMortalityMIMIC4``.

    Args:
        dataset: the SampleDataset to train on.
        num_layers: number of TPC layers. Default 5 (about 46 hours of
            history with kernel_size 4 and linear dilation).
        dilation: "linear" (1, 2, 3, ...) or "exponential" (1, 2, 4, ...).
        share_weights: share temporal kernels across features.
        use_temporal: enable temporal convolutions.
        use_pointwise: enable pointwise convolutions.
        use_skips: enable concatenated skip connections.
        temp_channels: temporal channels per feature.
        point_size: pointwise output size per layer.
        kernel_size: temporal kernel size.
        dropout: dropout rate.
        last_linear_size: hidden size of the prediction head.

    Examples:
        >>> from pyhealth.models import TPC
        >>> model = TPC(dataset=samples, num_layers=5)  # doctest: +SKIP
        >>> output = model(**batch)  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_layers: int = 5,
        dilation: str = "linear",
        share_weights: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skips: bool = True,
        temp_channels: int = 8,
        point_size: int = 16,
        kernel_size: int = 4,
        dropout: float = 0.3,
        last_linear_size: int = 32,
    ):
        super().__init__(dataset=dataset)
        # TODO: validate arguments, compute self.dilations, build an input
        # MaskedNorm1d, the stack of TPCLayers and the prediction head
        # (Linear -> ReLU -> Dropout -> Linear to self.get_output_size()).
        self.dilations: List[int] = []
        raise NotImplementedError

    @property
    def receptive_field(self) -> int:
        """Hours of history visible from the last hour.

        Formula: 1 + (kernel_size - 1) * sum(dilations), or 1 without the
        temporal branch.
        """
        # TODO
        raise NotImplementedError

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: a batch from ``get_dataloader``: each feature key maps to
                a [batch, hours, features] tensor, plus the label key.

        Returns:
            Dict with ``loss``, ``y_prob``, ``y_true`` and ``logit`` (and
            ``embed`` if ``embed=True`` is passed).
        """
        # TODO: mask = hours that are not all zero; normalize; run layers;
        # take features at each patient's last real hour; head -> logits;
        # loss = self.get_loss_function(); y_prob = self.prepare_y_prob().
        raise NotImplementedError
