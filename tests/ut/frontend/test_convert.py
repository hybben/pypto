# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure-Python unit tests for jit.convert(), parse_kernel_signature(), build_call_wrapper(),
_validate_args(), and _args_to_ctypes().

No hardware or external tools are required.
"""

import ctypes
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock torch before any pypto imports so jit.py's top-level 'import torch'
# resolves to our MagicMock.  torch.Tensor is replaced with a concrete class
# so that isinstance() checks inside _validate_args() work correctly.
# ---------------------------------------------------------------------------
_torch_mock = MagicMock()
sys.modules.setdefault("torch", _torch_mock)
sys.modules.setdefault("torch.npu", MagicMock())


class _MockTensor:
    """Minimal stand-in for torch.Tensor used in validation tests."""

    def __init__(self, shape: list[int], dtype):
        self.shape = shape
        self.dtype = dtype

    def data_ptr(self) -> int:
        return 0


# Assign the concrete class so isinstance(x, torch.Tensor) works in jit code.
_torch_mock.Tensor = _MockTensor

from pypto import DataType  # noqa: E402,I001
from pypto.frontend.jit import (  # noqa: E402
    ParamSpec,
    _args_to_ctypes,
    _collect_dyn_vars,
    _validate_args,
    build_call_wrapper,
    convert,
    parse_kernel_signature,
)
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# parse_kernel_signature
# ---------------------------------------------------------------------------


def test_parse_signature_pointer_only():
    """Regression: pointer-only params still produce is_ptr=True tuples."""
    line = "__global__ AICORE void my_kernel(__gm__ float* input, __gm__ float* output) {"
    result = parse_kernel_signature(line)
    assert result is not None
    name, params = result
    assert name == "my_kernel"
    assert params == [("float", "input", True), ("float", "output", True)]


def test_parse_signature_mixed_pointer_and_scalar():
    """New: scalar params (no pointer, no __gm__) produce is_ptr=False tuples."""
    line = "__global__ AICORE void scaled_kernel(__gm__ float* input, __gm__ float* output, int32_t n) {"
    result = parse_kernel_signature(line)
    assert result is not None
    name, params = result
    assert name == "scaled_kernel"
    assert params == [
        ("float", "input", True),
        ("float", "output", True),
        ("int32_t", "n", False),
    ]


def test_parse_signature_scalar_only():
    """A kernel with only scalar params."""
    line = "__global__ AICORE void scalar_kernel(int32_t a, float b) {"
    result = parse_kernel_signature(line)
    assert result is not None
    name, params = result
    assert name == "scalar_kernel"
    assert params == [("int32_t", "a", False), ("float", "b", False)]


def test_parse_signature_no_params():
    """A kernel with no params returns an empty list."""
    line = "__global__ AICORE void empty_kernel() {"
    result = parse_kernel_signature(line)
    assert result is not None
    name, params = result
    assert name == "empty_kernel"
    assert params == []


def test_parse_signature_multiline():
    """Multi-line signature (params split across lines) is handled."""
    line = "__global__ AICORE void multi_kernel("
    rest = "__gm__ half* x,\n    __gm__ half* y,\n    uint64_t offset) {"
    result = parse_kernel_signature(line, rest)
    assert result is not None
    name, params = result
    assert name == "multi_kernel"
    assert params == [
        ("half", "x", True),
        ("half", "y", True),
        ("uint64_t", "offset", False),
    ]


def test_parse_signature_no_match():
    """Non-kernel lines return None."""
    assert parse_kernel_signature("void not_a_kernel(int x) {") is None
    assert parse_kernel_signature("") is None


def test_parse_signature_plain_pointer_no_qualifier():
    """Pointer without __gm__ qualifier is NOT recognized (only __gm__ pointers are valid)."""
    line = "__global__ AICORE void plain_ptr_kernel(float* input, float* output) {"
    result = parse_kernel_signature(line)
    assert result is not None
    name, params = result
    assert name == "plain_ptr_kernel"
    # Without __gm__, these params are not matched as pointer or scalar → empty
    assert params == []


# ---------------------------------------------------------------------------
# build_call_wrapper
# ---------------------------------------------------------------------------


def test_build_wrapper_pointer_only():
    """Regression: pointer params are cast to their type in the wrapper."""
    params = [("float", "input", True), ("float", "output", True)]
    wrapper = build_call_wrapper("my_kernel", params)
    assert "uint8_t* input" in wrapper
    assert "uint8_t* output" in wrapper
    assert "(float *)input" in wrapper
    assert "(float *)output" in wrapper
    assert "my_kernel<<<blockDim, nullptr, stream>>>" in wrapper


def test_build_wrapper_mixed_params():
    """New: scalar params are passed by value (no cast, no uint8_t*)."""
    params = [("float", "input", True), ("float", "output", True), ("int32_t", "n", False)]
    wrapper = build_call_wrapper("scaled_kernel", params)
    # Pointer params
    assert "uint8_t* input" in wrapper
    assert "(float *)input" in wrapper
    assert "uint8_t* output" in wrapper
    assert "(float *)output" in wrapper
    # Scalar param: declared as its own type, passed directly
    assert "int32_t n" in wrapper
    assert "uint8_t* n" not in wrapper
    # The call should include n without a cast
    call_line = [line for line in wrapper.splitlines() if "scaled_kernel<<<" in line][0]
    assert "n" in call_line
    assert "(int32_t *)n" not in call_line


def test_build_wrapper_scalar_only():
    """Scalar-only params: no uint8_t* declarations, no casts."""
    params = [("int32_t", "a", False), ("float", "b", False)]
    wrapper = build_call_wrapper("scalar_kernel", params)
    assert "int32_t a" in wrapper
    assert "float b" in wrapper
    assert "uint8_t*" not in wrapper
    assert "scalar_kernel<<<blockDim, nullptr, stream>>>(a, b)" in wrapper


# ---------------------------------------------------------------------------
# convert() end-to-end
# ---------------------------------------------------------------------------

_POINTER_ONLY_CPP = """\
#include <cce/cce_aicore.hpp>

__global__ AICORE void my_kernel(__gm__ float* input, __gm__ float* output) {
    // body
}
"""

_MIXED_CPP = """\
#include <cce/cce_aicore.hpp>

__global__ AICORE void scaled_kernel(__gm__ float* input, __gm__ float* output, int32_t n) {
    // body
}
"""


def test_convert_appends_wrapper_pointer_only():
    """Regression: convert() appends a valid call_kernel for pointer-only kernels."""
    result = convert(_POINTER_ONLY_CPP)
    assert 'extern "C" void call_kernel(' in result
    assert "uint8_t* input" in result
    assert "uint8_t* output" in result
    assert "(float *)input" in result
    assert "(float *)output" in result
    assert "my_kernel<<<blockDim, nullptr, stream>>>" in result


def test_convert_appends_wrapper_mixed_params():
    """New: convert() emits correct wrapper for a kernel with pointer + scalar params."""
    result = convert(_MIXED_CPP)
    assert 'extern "C" void call_kernel(' in result
    # Pointer params
    assert "uint8_t* input" in result
    assert "uint8_t* output" in result
    assert "(float *)input" in result
    assert "(float *)output" in result
    # Scalar param by value
    assert "int32_t n" in result
    assert "uint8_t* n" not in result
    assert "scaled_kernel<<<blockDim, nullptr, stream>>>" in result
    # n must appear in the call without a cast
    call_line = [line for line in result.splitlines() if "scaled_kernel<<<" in line][0]
    assert ", n)" in call_line or call_line.rstrip().endswith(", n)")


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


def _tensor_spec(name, dtype, shape) -> ParamSpec:
    return ParamSpec(name, "tensor", dtype, shape)


def _ptr_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, "ptr", dtype, None)


def _scalar_spec(name, dtype) -> ParamSpec:
    return ParamSpec(name, "scalar", dtype, None)


def test_validate_args_count_mismatch():
    """Wrong number of args raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    with pytest.raises(TypeError, match="Expected 1 args, got 2"):
        _validate_args((_MockTensor([64], torch.float32), _MockTensor([64], torch.float32)), specs)


def test_validate_args_tensor_ok():
    """Correct tensor arg passes without error."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    _validate_args((_MockTensor([64], torch.float32),), specs)


def test_validate_args_tensor_wrong_type():
    """Non-tensor arg for a tensor param raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        _validate_args((3.14,), specs)


def test_validate_args_tensor_wrong_dtype():
    """Tensor with mismatched dtype raises TypeError."""
    specs = [_tensor_spec("x", DataType.INT32, [64])]
    tensor = _MockTensor([64], torch.float32)  # float32 != int32
    with pytest.raises(TypeError, match="dtype mismatch"):
        _validate_args((tensor,), specs)


def test_validate_args_tensor_wrong_rank():
    """Tensor with wrong number of dims raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64, 128])]
    tensor = _MockTensor([64], torch.float32)  # 1D instead of 2D
    with pytest.raises(TypeError, match="rank mismatch"):
        _validate_args((tensor,), specs)


def test_validate_args_tensor_wrong_static_dim():
    """Tensor where a static dim doesn't match raises TypeError."""
    specs = [_tensor_spec("x", DataType.FP32, [64, 128])]
    tensor = _MockTensor([64, 256], torch.float32)  # dim[1] is 256, expected 128
    with pytest.raises(TypeError, match="dim\\[1\\] mismatch"):
        _validate_args((tensor,), specs)


def test_validate_args_tensor_dynamic_dim_skipped():
    """Dynamic dims (−1) are not checked."""
    specs = [_tensor_spec("x", DataType.FP32, [-1, 128])]
    tensor = _MockTensor([999, 128], torch.float32)  # any size for dim 0
    _validate_args((tensor,), specs)  # must not raise


def test_validate_args_ptr_ok():
    """torch.Tensor is accepted for a ptr param."""
    specs = [_ptr_spec("p", DataType.FP32)]
    _validate_args((_MockTensor([1], torch.float32),), specs)


def test_validate_args_scalar_int_ok():
    """Python int passes for an integer scalar param."""
    specs = [_scalar_spec("n", DataType.INT32)]
    _validate_args((42,), specs)


def test_validate_args_scalar_float_ok():
    """Python float passes for a float scalar param."""
    specs = [_scalar_spec("scale", DataType.FP32)]
    _validate_args((1.5,), specs)


def test_validate_args_scalar_wrong_type():
    """Passing a tensor for a scalar param raises TypeError."""
    specs = [_scalar_spec("n", DataType.INT32)]
    with pytest.raises(TypeError, match="expected Python scalar"):
        _validate_args((_MockTensor([1], torch.float32),), specs)


def test_validate_args_scalar_float_for_int_dtype():
    """Passing float for an integer scalar dtype raises TypeError."""
    specs = [_scalar_spec("n", DataType.INT32)]
    with pytest.raises(TypeError, match="float value passed for non-float dtype"):
        _validate_args((1.5,), specs)


def test_validate_args_scalar_int_for_float_dtype():
    """Passing int for a float scalar dtype raises TypeError."""
    specs = [_scalar_spec("scale", DataType.FP32)]
    with pytest.raises(TypeError, match="int value passed for non-integer dtype"):
        _validate_args((42,), specs)


def test_validate_args_scalar_bool_for_int_dtype_ok():
    """bool is accepted for integer dtypes (bool is a subclass of int)."""
    specs = [_scalar_spec("flag", DataType.INT32)]
    _validate_args((True,), specs)  # must not raise


def test_validate_args_scalar_bool_for_bool_dtype_ok():
    """bool is accepted for BOOL dtype."""
    specs = [_scalar_spec("flag", DataType.BOOL)]
    _validate_args((False,), specs)  # must not raise


def test_validate_args_scalar_bool_for_float_dtype_raises():
    """bool is rejected for float dtypes (only valid for BOOL/integer dtypes)."""
    specs = [_scalar_spec("scale", DataType.FP32)]
    with pytest.raises(TypeError, match="bool value passed for non-boolean/non-integer dtype"):
        _validate_args((True,), specs)


def test_validate_args_mixed_params():
    """Mixed tensor + scalar params all validated correctly."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _tensor_spec("output", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    args = (
        _MockTensor([64], torch.float32),
        _MockTensor([64], torch.float32),
        128,
    )
    _validate_args(args, specs)  # must not raise


# ---------------------------------------------------------------------------
# _args_to_ctypes
# ---------------------------------------------------------------------------


def test_args_to_ctypes_tensor():
    """torch.Tensor → c_void_p wrapping data_ptr()."""
    specs = [_tensor_spec("x", DataType.FP32, [64])]
    tensor = _MockTensor([64], torch.float32)
    result = _args_to_ctypes((tensor,), specs)
    assert len(result) == 1
    assert isinstance(result[0], ctypes.c_void_p)
    assert result[0].value == tensor.data_ptr() or result[0].value is None  # 0 maps to None


def test_args_to_ctypes_ptr():
    """torch.Tensor for ptr param → c_void_p."""
    specs = [_ptr_spec("p", DataType.FP32)]
    tensor = _MockTensor([1], torch.float32)
    result = _args_to_ctypes((tensor,), specs)
    assert isinstance(result[0], ctypes.c_void_p)


def test_args_to_ctypes_scalar_float():
    """Python float → ctypes.c_float for FP32 param."""
    specs = [_scalar_spec("scale", DataType.FP32)]
    result = _args_to_ctypes((1.5,), specs)
    assert len(result) == 1
    assert isinstance(result[0], ctypes.c_float)
    assert abs(result[0].value - 1.5) < 1e-6


def test_args_to_ctypes_scalar_int():
    """Python int → ctypes.c_int32 for INT32 param."""
    specs = [_scalar_spec("n", DataType.INT32)]
    result = _args_to_ctypes((42,), specs)
    assert len(result) == 1
    assert isinstance(result[0], ctypes.c_int32)
    assert result[0].value == 42


def test_args_to_ctypes_mixed():
    """Mixed tensor + scalar → correct ctypes sequence."""
    specs = [
        _tensor_spec("input", DataType.FP32, [64]),
        _scalar_spec("n", DataType.INT32),
    ]
    tensor = _MockTensor([64], torch.float32)
    result = _args_to_ctypes((tensor, 7), specs)
    assert isinstance(result[0], ctypes.c_void_p)
    assert isinstance(result[1], ctypes.c_int32)
    assert result[1].value == 7


# ---------------------------------------------------------------------------
# _collect_dyn_vars
# ---------------------------------------------------------------------------


def test_collect_dyn_vars_basic():
    """Two tensors sharing M and N → [M, N] in appearance order."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["M", "N"]),
    ]
    assert _collect_dyn_vars(specs) == ["M", "N"]


def test_collect_dyn_vars_order():
    """Tensor a has M,N and tensor b has N,K → [M, N, K] (deduped, in order)."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["N", "K"]),
    ]
    assert _collect_dyn_vars(specs) == ["M", "N", "K"]


def test_collect_dyn_vars_no_dyn():
    """All-static shapes → empty list."""
    specs = [
        _tensor_spec("a", DataType.FP32, [64, 128]),
        _tensor_spec("b", DataType.FP32, [128]),
    ]
    assert _collect_dyn_vars(specs) == []


# ---------------------------------------------------------------------------
# _validate_args — dynamic variable tests
# ---------------------------------------------------------------------------


def test_validate_args_dyn_consistent():
    """M=64 in both tensors — consistent, must not raise."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([64, 64], torch.float32),
    )
    _validate_args(args, specs)  # must not raise


def test_validate_args_dyn_inconsistent():
    """M=64 in tensor a but M=128 in tensor b — must raise TypeError mentioning 'M'."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", 128]),
        _tensor_spec("b", DataType.FP32, ["M", 64]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128, 64], torch.float32),
    )
    with pytest.raises(TypeError, match="M"):
        _validate_args(args, specs)


def test_validate_args_static_dim_regression():
    """Static dim mismatch still raises (regression guard for the isinstance guard)."""
    specs = [_tensor_spec("x", DataType.FP32, [64, 128])]
    tensor = _MockTensor([64, 256], torch.float32)
    with pytest.raises(TypeError, match="dim\\[1\\] mismatch"):
        _validate_args((tensor,), specs)


def test_validate_args_minus1_skipped():
    """Old-style -1 (unnamed dynamic) is still skipped — backward compatible."""
    specs = [_tensor_spec("x", DataType.FP32, [-1, 64])]
    tensor = _MockTensor([999, 64], torch.float32)
    _validate_args((tensor,), specs)  # must not raise


# ---------------------------------------------------------------------------
# _args_to_ctypes — dynamic variable appending tests
# ---------------------------------------------------------------------------


def test_args_to_ctypes_dyn_appended():
    """Tensor with shape ["M", 64]: result should have c_void_p then c_int64(M_value)."""
    M = 32
    specs = [_tensor_spec("a", DataType.FP32, ["M", 64])]
    tensor = _MockTensor([M, 64], torch.float32)
    result = _args_to_ctypes((tensor,), specs)
    assert len(result) == 2
    assert isinstance(result[0], ctypes.c_void_p)
    assert isinstance(result[1], ctypes.c_int64)
    assert result[1].value == M


def test_args_to_ctypes_dyn_order():
    """Tensor a(M,N) and tensor b(N,K): appended order must be M, N, K."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["N", "K"]),
    ]
    args = (
        _MockTensor([10, 20], torch.float32),
        _MockTensor([20, 30], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    # 2 c_void_p + 3 c_int64 (M=10, N=20, K=30)
    assert len(result) == 5
    assert isinstance(result[2], ctypes.c_int64) and result[2].value == 10   # M
    assert isinstance(result[3], ctypes.c_int64) and result[3].value == 20   # N
    assert isinstance(result[4], ctypes.c_int64) and result[4].value == 30   # K


def test_args_to_ctypes_dyn_dedup():
    """Two tensors sharing M,N: each var appended exactly once."""
    specs = [
        _tensor_spec("a", DataType.FP32, ["M", "N"]),
        _tensor_spec("b", DataType.FP32, ["M", "N"]),
    ]
    args = (
        _MockTensor([8, 16], torch.float32),
        _MockTensor([8, 16], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    # 2 c_void_p + 2 c_int64 (M=8, N=16) — NOT 4 c_int64
    assert len(result) == 4
    assert result[2].value == 8    # M
    assert result[3].value == 16   # N


def test_args_to_ctypes_no_dyn():
    """No dynamic dims: result length equals number of args."""
    specs = [
        _tensor_spec("a", DataType.FP32, [64, 128]),
        _tensor_spec("b", DataType.FP32, [128]),
    ]
    args = (
        _MockTensor([64, 128], torch.float32),
        _MockTensor([128], torch.float32),
    )
    result = _args_to_ctypes(args, specs)
    assert len(result) == len(args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
