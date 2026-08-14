# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepGEMM linear warmup discovery.

Run `pytest tests/model_executor/test_deep_gemm_warmup.py`.
"""

from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.model_executor.warmup.deep_gemm_warmup import (
    _block_fp8_linear_kernel,
    _extract_data_from_linear_base_module,
)

BLOCK = [128, 128]


def _mock_layer(quant_method) -> Mock:
    """A LinearBase whose weights are already loaded and block-quantized."""
    layer = Mock(spec=LinearBase)
    layer.__class__ = LinearBase
    layer.quant_method = quant_method
    layer.weight = torch.empty((256, 512), dtype=torch.float8_e4m3fn)
    layer.weight_scale_inv = torch.empty((2, 4), dtype=torch.float32)
    # Set by every block-quantized linear method; the warmup reads it here
    # rather than from quant_config, which ModelOptLinearMethod does not have.
    layer.weight_block_size = BLOCK
    return layer


def _mock_fp8_linear_method(*, block_quant=True, use_marlin=False, kernel=None) -> Mock:
    method = Mock(spec=Fp8LinearMethod)
    method.__class__ = Fp8LinearMethod
    method.block_quant = block_quant
    method.use_marlin = use_marlin
    method.fp8_linear = kernel
    method.quant_config = Mock(weight_block_size=BLOCK)
    return method


def test_extract_block_size_matches_fp8_quant_config():
    """The warmup reads the block size off the layer instead of
    ``quant_method.quant_config.weight_block_size``, because the generic
    ModelOptLinearMethod has no quant_config. Fp8LinearMethod.create_weights
    assigns ``layer.weight_block_size = self.weight_block_size`` whenever
    block_quant is set, so the two must agree -- if that ever diverges, every
    block-FP8 model would warm up against the wrong shape.
    """
    method = _mock_fp8_linear_method()
    layer = _mock_layer(method)

    _, _, block_size = _extract_data_from_linear_base_module(layer)

    assert block_size == method.quant_config.weight_block_size


def test_extract_prefers_weight_scale_inv():
    layer = _mock_layer(_mock_fp8_linear_method())
    _, scale, _ = _extract_data_from_linear_base_module(layer)
    assert scale is layer.weight_scale_inv


@pytest.mark.parametrize(
    "block_quant, use_marlin, expected",
    [
        (True, False, True),  # block-quantized, non-marlin -> eligible
        (False, False, False),  # per-tensor fp8 has no block scales
        (True, True, False),  # marlin does not dispatch to DeepGEMM
    ],
)
def test_fp8_linear_method_kernel_gating(block_quant, use_marlin, expected):
    """Preserves the block_quant / use_marlin gate the isinstance-based
    discovery used to apply to Fp8LinearMethod."""
    kernel = Mock()
    method = _mock_fp8_linear_method(
        block_quant=block_quant, use_marlin=use_marlin, kernel=kernel
    )

    assert (_block_fp8_linear_kernel(method) is kernel) is expected


def test_modelopt_linear_method_exposes_its_kernel():
    """ModelOptLinearMethod keeps its kernel in ``kernel``, not ``fp8_linear``,
    and has no block_quant/use_marlin attributes -- so attribute-name-based
    discovery silently skipped FP8_PB / FP8_PB_WO layers."""
    method = Mock(spec=ModelOptLinearMethod)
    method.__class__ = ModelOptLinearMethod
    method.kernel = Mock()

    assert _block_fp8_linear_kernel(method) is method.kernel


def test_unknown_quant_method_is_not_eligible():
    assert _block_fp8_linear_kernel(Mock()) is None
    assert _block_fp8_linear_kernel(None) is None
