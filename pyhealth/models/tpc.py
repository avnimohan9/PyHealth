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

STATUS: implemented and unit-tested (tests/core/test_tpc.py).
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
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalizes x.

        Args:
            x: tensor of shape [batch, channels, hours].
            mask: bool tensor of shape [batch, hours]; True = real hour.

        Returns:
            Tensor of the same shape, with padded hours set to 0.
        """
        m = mask.unsqueeze(1).to(x.dtype)  # [batch, 1, hours]
        if self.training:
            count = m.sum()
            if count > 0:
                mean = (x * m).sum(dim=(0, 2)) / count
                var = (((x - mean[None, :, None]) ** 2) * m).sum(dim=(0, 2)) / count
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
            else:  # batch with no real hours: fall back to the running stats
                mean, var = self.running_mean, self.running_var
        else:
            mean, var = self.running_mean, self.running_var
        out = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * m


class TPCLayer(nn.Module):
    """One TPC layer: temporal convolution + pointwise convolution + skips.

    Channel layout of the layer input (and output), in this order:
        [raw features F | temporal outputs F * temp_channels | pointwise outputs point_size]
    A block is present only if it applies (the first layer receives just the
    raw features; raw features are carried forward only with ``use_skips``).

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
        first_layer: True for the first layer (its input is the raw features only).
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
        first_layer: bool = True,
    ):
        super().__init__()
        if not (use_temporal or use_pointwise):
            raise ValueError("At least one of use_temporal / use_pointwise must be True.")
        self.num_features = num_features
        self.temp_channels = temp_channels
        self.point_size = point_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.share_weights = share_weights
        self.use_temporal = use_temporal
        self.use_pointwise = use_pointwise
        self.use_skips = use_skips

        # What this layer receives.
        self.has_raw = first_layer or use_skips
        self.has_temp_in = (not first_layer) and use_temporal
        self.has_point_in = (not first_layer) and use_pointwise
        self.in_channels = (
            (num_features if self.has_raw else 0)
            + (num_features * temp_channels if self.has_temp_in else 0)
            + (point_size if self.has_point_in else 0)
        )
        # Channels each feature's temporal kernel sees: its raw value and
        # its own temporal channels from the previous layer.
        self.group_size = int(self.has_raw) + (temp_channels if self.has_temp_in else 0)
        # What this layer outputs (equals the next layer's in_channels).
        self.out_channels = (
            (num_features if use_skips else 0)
            + (num_features * temp_channels if use_temporal else 0)
            + (point_size if use_pointwise else 0)
        )

        if use_temporal:
            if share_weights:
                self.temporal = nn.Conv1d(
                    self.group_size, temp_channels, kernel_size, dilation=dilation
                )
            else:
                self.temporal = nn.Conv1d(
                    num_features * self.group_size,
                    num_features * temp_channels,
                    kernel_size,
                    dilation=dilation,
                    groups=num_features,
                )
            self.temporal_norm = MaskedNorm1d(num_features * temp_channels)
        if use_pointwise:
            self.pointwise = nn.Conv1d(self.in_channels, point_size, kernel_size=1)
            self.pointwise_norm = MaskedNorm1d(point_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Runs one layer.

        Args:
            x: layer input of shape [batch, channels, hours].
            mask: bool tensor of shape [batch, hours]; True = real hour.

        Returns:
            Input for the next layer, shape [batch, channels_out, hours].
        """
        batch, _, hours = x.shape
        feats = self.num_features
        outs = []

        if self.use_skips:
            outs.append(x[:, :feats])  # raw features, carried forward untouched

        if self.use_temporal:
            pieces = []
            if self.has_raw:
                pieces.append(x[:, :feats].unsqueeze(2))  # [batch, F, 1, hours]
            if self.has_temp_in:
                start = feats if self.has_raw else 0
                prev = x[:, start:start + feats * self.temp_channels]
                pieces.append(prev.reshape(batch, feats, self.temp_channels, hours))
            g = torch.cat(pieces, dim=2)  # [batch, F, group_size, hours]
            if self.share_weights:
                g = g.reshape(batch * feats, self.group_size, hours)
            else:
                g = g.reshape(batch, feats * self.group_size, hours)
            # Left-pad so the convolution is causal: hour t sees only hours <= t.
            g = nn.functional.pad(g, ((self.kernel_size - 1) * self.dilation, 0))
            y = self.temporal(g).reshape(batch, feats * self.temp_channels, hours)
            y = self.temporal_norm(y, mask)
            outs.append(self.dropout(torch.relu(y)))

        if self.use_pointwise:
            p = self.pointwise_norm(self.pointwise(x), mask)
            outs.append(self.dropout(torch.relu(p)))

        return torch.cat(outs, dim=1)


class TPC(BaseModel):
    """Temporal Pointwise Convolutional Network for timeseries inputs.

    The dataset must have exactly one ``timeseries`` input of shape
    [batch, hours, features], for example the ``labs`` key of
    ``InHospitalMortalityMIMIC4``. Padded hours are detected as hours where
    every feature is zero, which is safe for that task because every real
    hour has at least one measured lab.

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
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if dilation not in ("linear", "exponential"):
            raise ValueError("dilation must be 'linear' or 'exponential'.")
        if min(kernel_size, temp_channels, point_size, last_linear_size) < 1:
            raise ValueError("kernel_size, temp_channels, point_size and last_linear_size must be >= 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not (use_temporal or use_pointwise):
            raise ValueError("At least one of use_temporal / use_pointwise must be True.")
        if len(self.feature_keys) != 1:
            raise ValueError("TPC expects exactly one timeseries feature key.")
        if len(self.label_keys) != 1:
            raise ValueError("TPC expects exactly one label key.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.use_temporal = use_temporal
        if dilation == "linear":
            self.dilations: List[int] = list(range(1, num_layers + 1))
        else:
            self.dilations = [2 ** i for i in range(num_layers)]

        # Number of input features (labs) = last dimension of one sample.
        self.num_features = int(dataset[0][self.feature_key].shape[-1])

        self.input_norm = MaskedNorm1d(self.num_features)
        self.layers = nn.ModuleList(
            [
                TPCLayer(
                    num_features=self.num_features,
                    temp_channels=temp_channels,
                    point_size=point_size,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                    share_weights=share_weights,
                    use_temporal=use_temporal,
                    use_pointwise=use_pointwise,
                    use_skips=use_skips,
                    first_layer=(i == 0),
                )
                for i, d in enumerate(self.dilations)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.layers[-1].out_channels, last_linear_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_linear_size, self.get_output_size()),
        )

    @property
    def receptive_field(self) -> int:
        """Hours of history visible from the last hour.

        Formula: 1 + (kernel_size - 1) * sum(dilations), or 1 without the
        temporal branch.
        """
        if not self.use_temporal:
            return 1
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: a batch from ``get_dataloader``: each feature key maps to
                a [batch, hours, features] tensor, plus the label key.

        Returns:
            Dict with ``loss``, ``y_prob``, ``y_true`` and ``logit`` (and
            ``embed`` if ``embed=True`` is passed).
        """
        x = kwargs[self.feature_key].to(self.device).float()  # [batch, hours, F]
        mask = (x != 0).any(dim=-1)  # [batch, hours]; True = real hour

        h = self.input_norm(x.transpose(1, 2), mask)  # [batch, F, hours]
        for layer in self.layers:
            h = layer(h, mask)

        # Features at each patient's last real hour (0 if a row has none).
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
        last = (positions * mask).max(dim=1).values  # [batch]
        h = h[torch.arange(h.shape[0], device=h.device), :, last]  # [batch, channels]

        logits = self.head(h)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = h
        return results
