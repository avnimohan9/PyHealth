"""Temporal Pointwise Convolutional Networks (TPC).

Rocheteau, Liò & Hyland. "Temporal Pointwise Convolutional Networks for Length
of Stay Prediction in the Intensive Care Unit." CHIL 2021.
https://arxiv.org/abs/2007.09483

Ported from the authors' implementation
(https://github.com/EmmaRocheteau/TPC-LoS-prediction, ``models/tpc_model.py``)
to the PyHealth ``BaseModel`` interface. Differences from the original code:

    - Inputs are the F timeseries features of one PyHealth feature key; there
      are no decay-indicator mask channels, diagnoses or flat features.
    - BatchNorm is replaced by a masked per-channel normalization so padded
      hours do not enter the statistics and stay zero afterwards.
    - "linear" dilation uses d_i = i + 1 (1, 2, 3, ...). The authors' code uses
      d_0 = 1, d_i = i * (kernel_size - 1).
    - One prediction per patient, read from the last real hour.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class MaskedNorm1d(nn.Module):
    """Per-channel normalization over valid time steps only.

    Args:
        num_features: number of channels C.
        momentum: running-statistics momentum.
        affine: whether to learn a per-channel scale and shift.
        eps: numerical stability constant.

    Inputs are shaped [B, C, T] with a [B, T] boolean mask of valid steps.
    Padded steps are zero in the output.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1).to(x.dtype)  # [B, 1, T]
        if self.training:
            n = m.sum().clamp(min=1.0)
            mean = (x * m).sum(dim=(0, 2)) / n
            var = (((x - mean[None, :, None]) ** 2) * m).sum(dim=(0, 2)) / n
            with torch.no_grad():
                unbiased = var * n / (n - 1).clamp(min=1.0)
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(unbiased, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        out = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        if self.weight is not None:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out * m


class _BatchNorm1dIgnoringMask(nn.BatchNorm1d):
    """Plain BatchNorm1d: padded steps enter the statistics (for ablation)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class _NoNorm(nn.Module):
    """Identity normalization (for ablation)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x


def make_norm(kind: str, num_features: int, affine: bool = True) -> nn.Module:
    """Builds a normalization layer taking (x, mask).

    Args:
        kind: "masked" (statistics over valid steps only), "batch" (standard
            BatchNorm1d over all steps, padding included) or "none".
        num_features: number of channels.
        affine: whether to learn a scale and shift.
    """
    if kind == "masked":
        return MaskedNorm1d(num_features, affine=affine)
    if kind == "batch":
        return _BatchNorm1dIgnoringMask(num_features, affine=affine)
    if kind == "none":
        return _NoNorm()
    raise ValueError(f"Unknown norm {kind!r}.")


class TPCLayer(nn.Module):
    """One TPC layer: per-feature temporal convolution plus pointwise mixing.

    Args:
        num_features: F, the number of raw input features.
        prev_point_total: Zt, pointwise outputs accumulated by earlier layers.
        prev_point_size: Z, the previous layer's pointwise output size.
        prev_temp_channels: Y, the previous layer's temporal channels per
            feature (0 for the first layer).
        temp_channels: temporal output channels per feature.
        point_size: pointwise output size.
        kernel_size: temporal kernel size.
        dilation: temporal dilation.
        dropout: dropout rate.
        share_weights: share one temporal kernel across features.
        use_temporal: enable the temporal convolution.
        use_pointwise: enable the pointwise convolution.
        use_skips: concatenate the inputs and earlier outputs (the full model).
            If False, the temporal and pointwise branches only see their own
            previous outputs, as in the authors' no-skip variant.
        norm: normalization type, see ``make_norm``.
    """

    def __init__(
        self,
        num_features: int,
        prev_point_total: int,
        prev_point_size: int,
        prev_temp_channels: int,
        temp_channels: int,
        point_size: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        share_weights: bool = False,
        use_temporal: bool = True,
        use_pointwise: bool = True,
        use_skips: bool = True,
        norm: str = "masked",
    ):
        super().__init__()
        self.F = num_features
        self.Zt = prev_point_total
        self.Z = prev_point_size
        self.Y = prev_temp_channels
        self.k = temp_channels
        self.P = point_size
        self.share_weights = share_weights
        self.use_temporal = use_temporal
        self.use_pointwise = use_pointwise
        self.use_skips = use_skips
        self.padding = (kernel_size - 1) * dilation
        self.dropout = nn.Dropout(dropout)

        # Number of "features" the temporal kernels slide over, and the
        # channels each one carries in.
        if use_skips:
            self.groups = num_features + prev_point_total
            per_group_in = 1 + prev_temp_channels
        else:
            self.groups = num_features
            per_group_in = max(prev_temp_channels, 1)

        if use_temporal:
            if share_weights:
                self.temp = nn.Conv1d(
                    per_group_in, temp_channels, kernel_size, dilation=dilation
                )
            else:
                self.temp = nn.Conv1d(
                    self.groups * per_group_in,
                    self.groups * temp_channels,
                    kernel_size,
                    dilation=dilation,
                    groups=self.groups,
                )
            self.norm_temp = make_norm(norm, self.groups * temp_channels)

        if use_pointwise:
            if not use_skips:
                # Previous pointwise output, or the raw features in layer 0.
                point_in = prev_point_size if prev_point_size > 0 else num_features
            elif use_temporal:
                # Previous temporal output, previous pointwise output, raw input.
                point_in = (
                    (num_features + prev_point_total - prev_point_size)
                    * prev_temp_channels
                    + prev_point_size
                    + num_features
                )
            else:
                # Pointwise-only: raw input plus all earlier pointwise outputs.
                point_in = num_features + prev_point_total
            self.point = nn.Linear(point_in, point_size)
            self.norm_point = make_norm(norm, point_size)

    def _temporal(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B, groups * c_in, T] -> [B, groups * k, T]."""
        B, _, T = x.shape
        x = F.pad(x, (self.padding, 0))
        if self.share_weights:
            out = self.temp(x.reshape(B * self.groups, -1, x.shape[-1]))
            out = out.reshape(B, self.groups * self.k, T)
        else:
            out = self.temp(x)
        return self.dropout(self.norm_temp(out, mask))

    def _pointwise(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B, c_in, T] -> [B, P, T]."""
        out = self.point(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm_point(out, mask))

    def forward(
        self,
        x: torch.Tensor,
        x_orig: torch.Tensor,
        skip: Optional[torch.Tensor],
        prev_temp: Optional[torch.Tensor],
        prev_point: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Runs one layer.

        Args:
            x: temporal input, [B, groups * c_in, T].
            x_orig: raw input features, [B, F, T].
            skip: raw features plus earlier pointwise outputs, [B, F + Zt, T].
            prev_temp: previous temporal output, [B, (F + Zt - Z) * Y, T].
            prev_point: previous pointwise output, [B, Z, T].
            mask: valid time steps, [B, T].

        Returns:
            (next_x, skip, temp_out, point_out). ``next_x`` is the next
            layer's temporal input.
        """
        B, _, T = x_orig.shape
        temp_out = self._temporal(x, mask) if self.use_temporal else None

        if not self.use_skips:
            point_out = None
            if self.use_pointwise:
                point_in = prev_point if prev_point is not None else x_orig
                point_out = torch.relu(self._pointwise(point_in, mask))
            if temp_out is not None:
                temp_out = torch.relu(temp_out)
            return temp_out, skip, temp_out, point_out

        point_out = None
        if self.use_pointwise:
            if self.use_temporal:
                parts = [p for p in (prev_temp, prev_point) if p is not None]
                point_in = torch.cat(parts + [x_orig], dim=1)
            else:
                point_in = skip
            point_out = self._pointwise(point_in, mask)

        if not self.use_temporal:
            # Pointwise-only: grow the skip stack with this layer's output.
            skip = torch.cat([skip, point_out], dim=1)
            return torch.relu(skip), skip, None, point_out

        # Add the previous layer's pointwise output as new "features".
        if prev_point is not None:
            skip = torch.cat([skip, prev_point], dim=1)
        G = skip.shape[1]
        temp_skip = torch.cat(
            [skip.unsqueeze(2), temp_out.reshape(B, G, self.k, T)], dim=2
        )  # [B, G, 1 + k, T]
        if point_out is not None:
            point_rep = point_out.unsqueeze(2).expand(-1, -1, 1 + self.k, -1)
            temp_skip = torch.cat([temp_skip, point_rep], dim=1)
        next_x = torch.relu(temp_skip).reshape(B, -1, T)
        return next_x, skip, temp_out, point_out


class TPC(BaseModel):
    """Temporal Pointwise Convolutional Network for timeseries inputs.

    Each feature key must be a ``timeseries`` input of shape [B, T, F]. The
    model stacks TPC layers over the hourly grid and predicts from the last
    valid hour of each sequence.

    Args:
        dataset: the SampleDataset to train on.
        num_layers: number of TPC layers. Default 5.
        dilation: "linear" (d_i = i + 1) or "exponential" (d_i = 2 ** i).
        share_weights: share temporal kernels across features.
        use_temporal: enable temporal convolutions.
        use_pointwise: enable pointwise convolutions.
        use_skips: enable concatenated skip connections.
        temp_channels: temporal channels per feature.
        point_size: pointwise output size per layer.
        kernel_size: temporal kernel size.
        dropout: dropout rate.
        last_linear_size: hidden size of the prediction head.
        norm: "masked" (default), "batch" or "none"; applies to the input
            and to every layer.

    Examples:
        >>> model = TPC(dataset=samples, num_layers=5, dilation="linear")
        >>> out = model(**batch)  # doctest: +SKIP
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
        norm: str = "masked",
    ):
        super().__init__(dataset=dataset)
        if not (use_temporal or use_pointwise):
            raise ValueError("At least one of use_temporal/use_pointwise is needed.")
        if dilation not in ("linear", "exponential"):
            raise ValueError(f"Unknown dilation {dilation!r}.")
        assert len(self.label_keys) == 1, "TPC supports one label key."
        self.label_key = self.label_keys[0]
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.use_temporal = use_temporal
        self.use_skips = use_skips
        self.dilations: List[int] = [
            (i + 1) if dilation == "linear" else 2**i for i in range(num_layers)
        ]

        self.input_norm = nn.ModuleDict()
        self.layers = nn.ModuleDict()
        head_in = 0
        for key in self.feature_keys:
            n_feat = self.dataset.input_processors[key].size()
            self.input_norm[key] = make_norm(norm, n_feat, affine=False)
            layers = nn.ModuleList()
            Zt, Z, Y = 0, 0, 0
            for i in range(num_layers):
                layers.append(
                    TPCLayer(
                        num_features=n_feat,
                        prev_point_total=Zt,
                        prev_point_size=Z,
                        prev_temp_channels=Y,
                        temp_channels=temp_channels,
                        point_size=point_size,
                        kernel_size=kernel_size,
                        dilation=self.dilations[i],
                        dropout=dropout,
                        share_weights=share_weights,
                        use_temporal=use_temporal,
                        use_pointwise=use_pointwise,
                        use_skips=use_skips,
                        norm=norm,
                    )
                )
                if use_temporal:
                    Y = temp_channels
                if use_pointwise:
                    # Zt counts every pointwise output so far, including the
                    # one that joins the skip stack in the next layer.
                    if use_skips:
                        Zt += point_size
                    Z = point_size
            self.layers[key] = layers
            # Final per-hour feature size.
            if not use_skips:
                head_in += (n_feat * Y if use_temporal else 0) + (
                    Z if use_pointwise else 0
                )
            elif use_temporal:
                head_in += (n_feat + Zt) * (1 + Y)
            else:
                head_in += n_feat + Zt

        self.head = nn.Sequential(
            nn.Linear(head_in, last_linear_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_linear_size, self.get_output_size()),
        )

    @property
    def receptive_field(self) -> int:
        """Hours of history visible from the last step (current hour included)."""
        if not self.use_temporal:
            return 1
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def _encode(self, key: str, x: torch.Tensor) -> torch.Tensor:
        """[B, T, F] -> [B, D] features at the last valid hour."""
        x = x.to(self.device).float()
        mask = x.abs().sum(dim=-1) != 0  # [B, T]
        x_orig = self.input_norm[key](x.transpose(1, 2), mask)  # [B, F, T]

        next_x, skip = x_orig, x_orig
        temp_out = point_out = None
        for layer in self.layers[key]:
            next_x, skip, temp_out, point_out = layer(
                next_x, x_orig, skip, temp_out, point_out, mask
            )

        if not self.use_skips:
            parts = [p for p in (temp_out, point_out) if p is not None]
            feats = torch.cat(parts, dim=1)
        else:
            feats = next_x
        lengths = mask.sum(dim=1).clamp(min=1)
        # Padding is at the end, so the last valid hour is lengths - 1.
        idx = torch.arange(x.shape[0], device=x.device)
        return feats[idx, :, lengths - 1]

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Returns:
            dict with loss, y_prob, y_true and logit (and embed if
            ``embed=True`` is passed).
        """
        emb = []
        for key in self.feature_keys:
            x = kwargs[key]
            if isinstance(x, (tuple, list)):
                x = x[0]
            emb.append(self._encode(key, x))
        patient_emb = torch.cat(emb, dim=1)
        logits = self.head(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        results = {
            "loss": loss,
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
