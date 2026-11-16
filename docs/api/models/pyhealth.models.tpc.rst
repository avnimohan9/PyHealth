pyhealth.models.tpc
===================================

Temporal Pointwise Convolutional Network (TPC) for in-hospital mortality
prediction from hourly lab time series, adapted from Rocheteau, Liò and Hyland,
`"Temporal Pointwise Convolutional Networks for Length of Stay Prediction in
the Intensive Care Unit" <https://arxiv.org/abs/2007.09483>`_ (CHIL 2021).

Each TPC layer runs a grouped, causal, dilated temporal convolution (one set of
weights per lab) and a pointwise (1x1) convolution side by side, and
concatenates their outputs with the layer's inputs (skip connections). The
prediction is made from each patient's last real (non-padded) hour.

.. autoclass:: pyhealth.models.tpc.MaskedNorm1d
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.tpc.TPCLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TPC
    :members:
    :undoc-members:
    :show-inheritance:
