# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Consolidated auto (SSA) operations for PyPTO Language DSL.

This module merges the former block_ops, tensor_ops, and unified_ops into a
single file.  It provides:

- Block (Tile) operations that return new Tile values (SSA style)
- Tensor operations that return new Tensor values
- Unified dispatch functions that auto-select between tensor and block paths
  based on the input type (Tensor vs Tile)
- ``block`` and ``tensor`` namespace objects for explicit access
  (``block.add``, ``tensor.add``, etc.)
"""

import types as _types
from collections.abc import Sequence
from typing import Literal, Optional, TypeVar, Union, overload

from pypto.ir.op import block_ops as _ir_block_ops
from pypto.ir.op import tensor_ops as _ir_tensor_ops
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Expr, MemorySpace

from ....typing import IntLike, Scalar, Tensor, Tile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def _unwrap_rhs(rhs: int | float | Tensor | Scalar) -> int | float | Expr:
    """Unwrap rhs operand: extract Expr from Tensor/Scalar wrappers, pass through primitives."""
    if isinstance(rhs, (Tensor, Scalar)):
        return rhs.unwrap()
    return rhs


# ###########################################################################
#  BLOCK (Tile) OPERATIONS
# ###########################################################################

def make_tile(
    shape: Sequence[IntLike],
    dtype: DataType,
    target_memory: MemorySpace = MemorySpace.Vec,
    addr: Optional[Union[int, Expr]] = None,
    size: Optional[int] = None,
) -> Tile:
    call_expr = _ir_block_ops.make_tile(
        _normalize_intlike(shape), dtype, target_memory, addr, size
    )
    return Tile(expr=call_expr)


def load(
    tensor: Tensor,
    offsets: Sequence[IntLike],
    shapes: Sequence[IntLike],
    target_memory: MemorySpace = MemorySpace.Vec,
    valid_shapes: Sequence[IntLike] | None = None,
) -> Tile:
    if valid_shapes is None:
        valid_shapes = shapes
    call_expr = _ir_block_ops.load(
        tensor.unwrap(),
        _normalize_intlike(offsets),
        _normalize_intlike(shapes),
        _normalize_intlike(valid_shapes),
        target_memory,
    )
    return Tile(expr=call_expr)


def store(
    tile: Tile,
    offsets: Sequence[IntLike],
    shapes: Sequence[IntLike],
    output_tensor: Tensor,
) -> Tensor:
    call_expr = _ir_block_ops.store(
        tile.unwrap(), _normalize_intlike(offsets), _normalize_intlike(shapes), output_tensor.unwrap()
    )
    return Tensor(expr=call_expr)


def l0c_store(
    tile: Tile,
    offsets: Sequence[IntLike],
    shapes: Sequence[IntLike],
    output_tensor: Tensor,
) -> Tensor:
    call_expr = _ir_block_ops.l0c_store(
        tile.unwrap(), _normalize_intlike(offsets), _normalize_intlike(shapes), output_tensor.unwrap()
    )
    return Tensor(expr=call_expr)


def move(tile: Tile, target_memory: MemorySpace, transpose: bool = False) -> Tile:
    call_expr = _ir_block_ops.move(tile.unwrap(), target_memory, transpose)
    return Tile(expr=call_expr)


def vec_move(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.vec_move(tile.unwrap())
    return Tile(expr=call_expr)


def full(shape: list[int], dtype: DataType, value: int | float) -> Tile:
    call_expr = _ir_block_ops.full(shape, dtype, value)
    return Tile(expr=call_expr)


def fillpad(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.fillpad(tile.unwrap())
    return Tile(expr=call_expr)


def get_block_idx() -> Scalar:
    call_expr = _ir_block_ops.get_block_idx()
    return Scalar(expr=call_expr)


def get_subblock_idx() -> Scalar:
    call_expr = _ir_block_ops.get_subblock_idx()
    return Scalar(expr=call_expr)


def get_block_num() -> Scalar:
    call_expr = _ir_block_ops.get_block_num()
    return Scalar(expr=call_expr)


def index_cast(idx: int | float | Expr | Scalar) -> Scalar:
    idx_expr = idx.unwrap() if isinstance(idx, Scalar) else idx
    call_expr = _ir_block_ops.index_cast(idx_expr)
    return Scalar(expr=call_expr)


# --- Block: overlapping names (will be saved as _block_* references) ---

def add(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.add(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def sub(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.sub(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def mul(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.mul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def div(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.div(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def exp(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.exp(tile.unwrap())
    return Tile(expr=call_expr)


def cast(
    tile: Tile,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> Tile:
    call_expr = _ir_block_ops.cast(tile.unwrap(), target_type, mode)
    return Tile(expr=call_expr)


def matmul(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.matmul(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def row_max(tile: Tile, tmp_tile: Tile) -> Tile:
    call_expr = _ir_block_ops.row_max(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_sum(tile: Tile, tmp_tile: Tile) -> Tile:
    call_expr = _ir_block_ops.row_sum(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def maximum(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.maximum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def reshape(tile: Tile, shape: Sequence[IntLike]) -> Tile:
    tile_expr = tile.unwrap()
    call_expr = _ir_block_ops.reshape(tile_expr, _normalize_intlike(shape))
    return Tile(expr=call_expr)


def transpose(tile: Tile, axis1: int, axis2: int) -> Tile:
    tile_expr = tile.unwrap()
    call_expr = _ir_block_ops.transpose(tile_expr, axis1, axis2)
    return Tile(expr=call_expr)


def view(tile: Tile, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tile:
    tile_expr = tile.unwrap()
    call_expr = _ir_block_ops.view(tile_expr, _normalize_intlike(shape), _normalize_intlike(offset))
    return Tile(expr=call_expr)


def getval(tile: Tile, index: int | Expr | Scalar) -> Scalar:
    tile_expr = tile.unwrap()
    index_expr = index.unwrap() if isinstance(index, Scalar) else index
    call_expr = _ir_block_ops.getval(tile_expr, index_expr)
    return Scalar(expr=call_expr)


def setval(tile: Tile, index: int | Expr | Scalar, value: int | float | Expr | Scalar) -> Tile:
    tile_expr = tile.unwrap()
    index_expr = index.unwrap() if isinstance(index, Scalar) else index
    value_expr = value.unwrap() if isinstance(value, Scalar) else value
    call_expr = _ir_block_ops.setval(tile_expr, index_expr, value_expr)
    return Tile(expr=call_expr)


# Save block versions before unified dispatch overwrites them
_block_add = add
_block_sub = sub
_block_mul = mul
_block_div = div
_block_exp = exp
_block_cast = cast
_block_matmul = matmul
_block_row_max = row_max
_block_row_sum = row_sum
_block_maximum = maximum
_block_reshape = reshape
_block_transpose = transpose
_block_view = view
_block_getval = getval
_block_setval = setval

# --- Block: non-overlapping ops (continued) ---

def adds(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.adds(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def subs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.subs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def muls(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.muls(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def divs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.divs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def neg(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.neg(tile.unwrap())
    return Tile(expr=call_expr)


def sqrt(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.sqrt(tile.unwrap())
    return Tile(expr=call_expr)


def rsqrt(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.rsqrt(tile.unwrap())
    return Tile(expr=call_expr)


def recip(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.recip(tile.unwrap())
    return Tile(expr=call_expr)


def log(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.log(tile.unwrap())
    return Tile(expr=call_expr)


def abs(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.abs(tile.unwrap())
    return Tile(expr=call_expr)


def relu(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.relu(tile.unwrap())
    return Tile(expr=call_expr)


def matmul_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.matmul_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def matmul_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    call_expr = _ir_block_ops.matmul_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
    return Tile(expr=call_expr)


def gemv(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.gemv(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_acc(acc: Tile, lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.gemv_acc(acc.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def gemv_bias(lhs: Tile, rhs: Tile, bias: Tile) -> Tile:
    call_expr = _ir_block_ops.gemv_bias(lhs.unwrap(), rhs.unwrap(), bias.unwrap())
    return Tile(expr=call_expr)


def row_min(tile: Tile, tmp_tile: Tile) -> Tile:
    call_expr = _ir_block_ops.row_min(tile.unwrap(), tmp_tile.unwrap())
    return Tile(expr=call_expr)


def row_expand(src: Tile) -> Tile:
    call_expr = _ir_block_ops.row_expand(src.unwrap())
    return Tile(expr=call_expr)


def row_expand_sub(tile: Tile, row_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.row_expand_sub(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_div(tile: Tile, row_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.row_expand_div(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_mul(tile: Tile, row_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.row_expand_mul(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def row_expand_add(tile: Tile, row_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.row_expand_add(tile.unwrap(), row_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand(target: Tile, col_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.col_expand(target.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_mul(tile: Tile, col_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.col_expand_mul(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_div(tile: Tile, col_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.col_expand_div(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def col_expand_sub(tile: Tile, col_vec: Tile) -> Tile:
    call_expr = _ir_block_ops.col_expand_sub(tile.unwrap(), col_vec.unwrap())
    return Tile(expr=call_expr)


def expands(target: Tile, scalar: int | float | Expr | Scalar) -> Tile:
    scalar_expr = scalar.unwrap() if isinstance(scalar, Scalar) else scalar
    call_expr = _ir_block_ops.expands(target.unwrap(), scalar_expr)
    return Tile(expr=call_expr)


def minimum(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.minimum(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def cmp(lhs: Tile, rhs: Tile, cmp_type: int = 0) -> Tile:
    call_expr = _ir_block_ops.cmp(lhs.unwrap(), rhs.unwrap(), cmp_type)
    return Tile(expr=call_expr)


def cmps(lhs: Tile, rhs: int | float | Expr | Scalar, cmp_type: int = 0) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.cmps(lhs.unwrap(), rhs_expr, cmp_type)
    return Tile(expr=call_expr)


def sum(tile: Tile, axis: int, keepdim: bool = False) -> Tile:
    call_expr = _ir_block_ops.sum(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


@overload
def max(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...
@overload
def max(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...
def max(tile: Tile | Scalar, axis: int | Scalar = 0, keepdim: bool = False) -> Tile | Scalar:
    if isinstance(tile, Scalar):
        rhs: Expr = (
            axis.unwrap()
            if isinstance(axis, Scalar)
            else _ir_core.ConstInt(axis, DataType.INT32, _ir_core.Span.unknown())
        )
        return Scalar(expr=_ir_core.max_(tile.unwrap(), rhs))
    assert isinstance(axis, int)
    call_expr = _ir_block_ops.max(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


@overload
def min(tile: Tile, axis: int, keepdim: bool = False) -> Tile: ...
@overload
def min(tile: Scalar, axis: Scalar | int, keepdim: bool = False) -> Scalar: ...
def min(tile: Tile | Scalar, axis: int | Scalar = 0, keepdim: bool = False) -> Tile | Scalar:
    if isinstance(tile, Scalar):
        rhs: Expr = (
            axis.unwrap()
            if isinstance(axis, Scalar)
            else _ir_core.ConstInt(axis, DataType.INT32, _ir_core.Span.unknown())
        )
        return Scalar(expr=_ir_core.min_(tile.unwrap(), rhs))
    assert isinstance(axis, int)
    call_expr = _ir_block_ops.min(tile.unwrap(), axis, keepdim)
    return Tile(expr=call_expr)


def rem(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.rem(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def rems(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.rems(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def and_(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.and_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ands(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.ands(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def or_(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.or_(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def ors(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.ors(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def xor(lhs: Tile, rhs: Tile, tmp: Tile) -> Tile:
    call_expr = _ir_block_ops.xor(lhs.unwrap(), rhs.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def xors(lhs: Tile, rhs: int | Expr | Scalar, tmp: Tile) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.xors(lhs.unwrap(), rhs_expr, tmp.unwrap())
    return Tile(expr=call_expr)


def shl(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.shl(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shls(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.shls(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def shr(lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.shr(lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def shrs(lhs: Tile, rhs: int | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.shrs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def maxs(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.maxs(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def mins(lhs: Tile, rhs: int | float | Expr | Scalar) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.mins(lhs.unwrap(), rhs_expr)
    return Tile(expr=call_expr)


def prelu(tile: Tile, slope: Tile, tmp: Tile) -> Tile:
    call_expr = _ir_block_ops.prelu(tile.unwrap(), slope.unwrap(), tmp.unwrap())
    return Tile(expr=call_expr)


def not_(tile: Tile) -> Tile:
    call_expr = _ir_block_ops.not_(tile.unwrap())
    return Tile(expr=call_expr)


def addc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    call_expr = _ir_block_ops.addc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def subc(lhs: Tile, rhs: Tile, rhs2: Tile) -> Tile:
    call_expr = _ir_block_ops.subc(lhs.unwrap(), rhs.unwrap(), rhs2.unwrap())
    return Tile(expr=call_expr)


def addsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.addsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def subsc(lhs: Tile, rhs: int | float | Expr | Scalar, rhs2: Tile) -> Tile:
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_block_ops.subsc(lhs.unwrap(), rhs_expr, rhs2.unwrap())
    return Tile(expr=call_expr)


def lrelu(tile: Tile, slope: int | float | Expr | Scalar) -> Tile:
    slope_expr = slope.unwrap() if isinstance(slope, Scalar) else slope
    call_expr = _ir_block_ops.lrelu(tile.unwrap(), slope_expr)
    return Tile(expr=call_expr)


def sel(mask: Tile, lhs: Tile, rhs: Tile) -> Tile:
    call_expr = _ir_block_ops.sel(mask.unwrap(), lhs.unwrap(), rhs.unwrap())
    return Tile(expr=call_expr)


def sels(lhs: Tile, rhs: Tile, select_mode: int | float | Expr | Scalar) -> Tile:
    select_mode_expr = select_mode.unwrap() if isinstance(select_mode, Scalar) else select_mode
    call_expr = _ir_block_ops.sels(lhs.unwrap(), rhs.unwrap(), select_mode_expr)
    return Tile(expr=call_expr)


# ###########################################################################
#  TENSOR OPERATIONS
# ###########################################################################

def create_tensor(shape: Sequence[IntLike], dtype: DataType) -> Tensor:
    call_expr = _ir_tensor_ops.create(_normalize_intlike(shape), dtype)
    return Tensor(expr=call_expr)


def read(tensor: Tensor, indices: Sequence[IntLike]) -> Scalar:
    tensor_expr = tensor.unwrap()
    call_expr = _ir_tensor_ops.read(tensor_expr, _normalize_intlike(indices))
    return Scalar(expr=call_expr)


def dim(tensor: Tensor, axis: int) -> Scalar:
    tensor_expr = tensor.unwrap()
    call_expr = _ir_tensor_ops.dim(tensor_expr, axis)
    return Scalar(expr=call_expr)


def _tensor_view(tensor: Tensor, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> Tensor:
    tensor_expr = tensor.unwrap()
    call_expr = _ir_tensor_ops.view(tensor_expr, _normalize_intlike(shape), _normalize_intlike(offset))
    return Tensor(expr=call_expr)


def _tensor_matmul(
    lhs: Tensor,
    rhs: Tensor,
    out_dtype: int | DataType | None = None,
    a_trans: bool = False,
    b_trans: bool = False,
    c_matrix_nz: bool = False,
) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = _ir_tensor_ops.matmul(lhs_expr, rhs_expr, out_dtype, a_trans, b_trans, c_matrix_nz)
    return Tensor(expr=call_expr)


def _tensor_mul(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_tensor_ops.mul(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def mul_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    lhs_expr = lhs.unwrap()
    call_expr = _ir_tensor_ops.mul_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def _tensor_add(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_tensor_ops.add(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def add_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    lhs_expr = lhs.unwrap()
    call_expr = _ir_tensor_ops.add_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def _tensor_sub(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_tensor_ops.sub(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def sub_scalar(lhs: Tensor, rhs: int | float | Expr) -> Tensor:
    lhs_expr = lhs.unwrap()
    call_expr = _ir_tensor_ops.sub_scalar(lhs_expr, rhs)
    return Tensor(expr=call_expr)


def _tensor_div(lhs: Tensor, rhs: int | float | Tensor | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = _unwrap_rhs(rhs)
    call_expr = _ir_tensor_ops.div(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def div_scalar(lhs: Tensor, rhs: int | float | Expr | Scalar) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap() if isinstance(rhs, Scalar) else rhs
    call_expr = _ir_tensor_ops.div_scalar(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def _tensor_maximum(lhs: Tensor, rhs: Tensor) -> Tensor:
    lhs_expr = lhs.unwrap()
    rhs_expr = rhs.unwrap()
    call_expr = _ir_tensor_ops.maximum(lhs_expr, rhs_expr)
    return Tensor(expr=call_expr)


def _tensor_row_max(input: Tensor) -> Tensor:
    input_expr = input.unwrap()
    call_expr = _ir_tensor_ops.row_max(input_expr)
    return Tensor(expr=call_expr)


def _tensor_row_sum(input: Tensor) -> Tensor:
    input_expr = input.unwrap()
    call_expr = _ir_tensor_ops.row_sum(input_expr)
    return Tensor(expr=call_expr)


def _tensor_exp(input: Tensor) -> Tensor:
    input_expr = input.unwrap()
    call_expr = _ir_tensor_ops.exp(input_expr)
    return Tensor(expr=call_expr)


def _tensor_cast(
    input: Tensor,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> Tensor:
    input_expr = input.unwrap()
    call_expr = _ir_tensor_ops.cast(input_expr, target_type, mode)
    return Tensor(expr=call_expr)


def assemble(target: Tensor, source: Tensor, offset: Sequence[IntLike]) -> Tensor:
    target_expr = target.unwrap()
    source_expr = source.unwrap()
    call_expr = _ir_tensor_ops.assemble(target_expr, source_expr, _normalize_intlike(offset))
    return Tensor(expr=call_expr)


def _tensor_reshape(tensor: Tensor, shape: Sequence[IntLike]) -> Tensor:
    tensor_expr = tensor.unwrap()
    call_expr = _ir_tensor_ops.reshape(tensor_expr, _normalize_intlike(shape))
    return Tensor(expr=call_expr)


def _tensor_transpose(tensor: Tensor, axis1: int, axis2: int) -> Tensor:
    tensor_expr = tensor.unwrap()
    call_expr = _ir_tensor_ops.transpose(tensor_expr, axis1, axis2)
    return Tensor(expr=call_expr)


def _tensor_getval(tensor: Tensor, offset: int | Expr | Scalar) -> Scalar:
    tensor_expr = tensor.unwrap()
    offset_expr = offset.unwrap() if isinstance(offset, Scalar) else offset
    call_expr = _ir_tensor_ops.getval(tensor_expr, offset_expr)
    return Scalar(expr=call_expr)


def _tensor_setval(tensor: Tensor, offset: int | Expr | Scalar, value: int | float | Expr | Scalar) -> Tensor:
    tensor_expr = tensor.unwrap()
    offset_expr = offset.unwrap() if isinstance(offset, Scalar) else offset
    value_expr = value.unwrap() if isinstance(value, Scalar) else value
    call_expr = _ir_tensor_ops.setval(tensor_expr, offset_expr, value_expr)
    return Tensor(expr=call_expr)


# ###########################################################################
#  UNIFIED DISPATCH (overwrites block versions for overlapping names)
# ###########################################################################

T = TypeVar("T", Tensor, Tile)


def add(lhs: T, rhs: T | int | float | Scalar) -> T:  # noqa: F811
    """Element-wise addition, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_add(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return adds(lhs, rhs)
    raise TypeError(f"add: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def sub(lhs: T, rhs: T | int | float | Scalar) -> T:  # noqa: F811
    """Element-wise subtraction, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_sub(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return subs(lhs, rhs)
    raise TypeError(f"sub: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def mul(lhs: T, rhs: T | int | float | Scalar) -> T:  # noqa: F811
    """Element-wise multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_mul(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return muls(lhs, rhs)
    raise TypeError(f"mul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def div(lhs: T, rhs: T | int | float | Scalar) -> T:  # noqa: F811
    """Element-wise division, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, (Tensor, int, float, Scalar)):
        return _tensor_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_div(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, (int, float, Scalar)):
        return divs(lhs, rhs)
    raise TypeError(f"div: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def maximum(lhs: T, rhs: T) -> T:  # noqa: F811
    """Element-wise maximum, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor_maximum(lhs, rhs)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_maximum(lhs, rhs)
    raise TypeError(f"maximum: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def exp(input: T) -> T:  # noqa: F811
    """Element-wise exponential, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_exp(input)
    if isinstance(input, Tile):
        return _block_exp(input)
    raise TypeError(f"exp: expected Tensor or Tile, got {type(input).__name__}")


def reshape(input: T, shape: Sequence[IntLike]) -> T:  # noqa: F811
    """Reshape operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_reshape(input, shape)
    if isinstance(input, Tile):
        return _block_reshape(input, shape)
    raise TypeError(f"reshape: expected Tensor or Tile, got {type(input).__name__}")


def transpose(input: T, axis1: int, axis2: int) -> T:  # noqa: F811
    """Transpose operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_transpose(input, axis1, axis2)
    if isinstance(input, Tile):
        return _block_transpose(input, axis1, axis2)
    raise TypeError(f"transpose: expected Tensor or Tile, got {type(input).__name__}")


def view(input: T, shape: Sequence[IntLike], offset: Sequence[IntLike]) -> T:  # noqa: F811
    """View/slice operation, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_view(input, shape, offset)
    if isinstance(input, Tile):
        return _block_view(input, shape, offset)
    raise TypeError(f"view: expected Tensor or Tile, got {type(input).__name__}")


@overload
def matmul(  # noqa: F811
    lhs: Tensor, rhs: Tensor,
    out_dtype: int | DataType | None = ..., a_trans: bool = ...,
    b_trans: bool = ..., c_matrix_nz: bool = ...,
) -> Tensor: ...
@overload
def matmul(lhs: Tile, rhs: Tile) -> Tile: ...  # noqa: F811
def matmul(  # noqa: F811
    lhs: T, rhs: T,
    out_dtype: int | DataType | None = None, a_trans: bool = False,
    b_trans: bool = False, c_matrix_nz: bool = False,
) -> T:
    """Matrix multiplication, dispatched by input type."""
    if isinstance(lhs, Tensor) and isinstance(rhs, Tensor):
        return _tensor_matmul(lhs, rhs, out_dtype, a_trans, b_trans, c_matrix_nz)
    if isinstance(lhs, Tile) and isinstance(rhs, Tile):
        return _block_matmul(lhs, rhs)
    raise TypeError(f"matmul: expected Tensor or Tile for lhs, got {type(lhs).__name__}")


def row_max(input: T, tmp_tile: Tile | None = None) -> T:  # noqa: F811
    """Row-wise max reduction, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_row_max(input)
    if isinstance(input, Tile):
        if tmp_tile is None:
            raise ValueError("row_max on Tile requires tmp_tile argument")
        return _block_row_max(input, tmp_tile)
    raise TypeError(f"row_max: expected Tensor or Tile, got {type(input).__name__}")


def row_sum(input: T, tmp_tile: Tile | None = None) -> T:  # noqa: F811
    """Row-wise sum reduction, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_row_sum(input)
    if isinstance(input, Tile):
        if tmp_tile is None:
            raise ValueError("row_sum on Tile requires tmp_tile argument")
        return _block_row_sum(input, tmp_tile)
    raise TypeError(f"row_sum: expected Tensor or Tile, got {type(input).__name__}")


def cast(  # noqa: F811
    input: T,
    target_type: int | DataType,
    mode: Literal["none", "rint", "round", "floor", "ceil", "trunc", "odd"] = "round",
) -> T:
    """Type casting, dispatched by input type."""
    if isinstance(input, Tensor):
        return _tensor_cast(input, target_type, mode)
    if isinstance(input, Tile):
        return _block_cast(input, target_type, mode)
    raise TypeError(f"cast: expected Tensor or Tile, got {type(input).__name__}")


def getval(input, offset):  # noqa: F811
    """Get scalar value, dispatched by input type (Tile or Tensor)."""
    if isinstance(input, Tensor):
        return _tensor_getval(input, offset)
    if isinstance(input, Tile):
        return _block_getval(input, offset)
    raise TypeError(f"getval: expected Tensor or Tile, got {type(input).__name__}")


def setval(input, offset, value):  # noqa: F811
    """Set scalar value, dispatched by input type (Tile or Tensor)."""
    if isinstance(input, Tensor):
        return _tensor_setval(input, offset, value)
    if isinstance(input, Tile):
        return _block_setval(input, offset, value)
    raise TypeError(f"setval: expected Tensor or Tile, got {type(input).__name__}")


# ###########################################################################
#  NAMESPACE OBJECTS (for pl.block.* and pl.tensor.* access)
# ###########################################################################

block = _types.SimpleNamespace(
    # Overlapping (block versions)
    add=_block_add, sub=_block_sub, mul=_block_mul, div=_block_div,
    exp=_block_exp, cast=_block_cast, matmul=_block_matmul,
    row_max=_block_row_max, row_sum=_block_row_sum, maximum=_block_maximum,
    reshape=_block_reshape, transpose=_block_transpose, view=_block_view,
    getval=_block_getval, setval=_block_setval,
    # Non-overlapping
    make_tile=make_tile, load=load, store=store, l0c_store=l0c_store,
    move=move, vec_move=vec_move, full=full, fillpad=fillpad,
    get_block_idx=get_block_idx, get_subblock_idx=get_subblock_idx,
    get_block_num=get_block_num, index_cast=index_cast,
    adds=adds, subs=subs, muls=muls, divs=divs,
    neg=neg, sqrt=sqrt, rsqrt=rsqrt, recip=recip, log=log,
    abs=abs, relu=relu, not_=not_,
    matmul_acc=matmul_acc, matmul_bias=matmul_bias,
    gemv=gemv, gemv_acc=gemv_acc, gemv_bias=gemv_bias,
    row_min=row_min, minimum=minimum,
    row_expand=row_expand, row_expand_sub=row_expand_sub,
    row_expand_div=row_expand_div, row_expand_mul=row_expand_mul,
    row_expand_add=row_expand_add,
    col_expand=col_expand, col_expand_mul=col_expand_mul,
    col_expand_div=col_expand_div, col_expand_sub=col_expand_sub,
    expands=expands, cmp=cmp, cmps=cmps,
    sum=sum, max=max, min=min,
    rem=rem, rems=rems,
    and_=and_, ands=ands, or_=or_, ors=ors,
    xor=xor, xors=xors, shl=shl, shls=shls, shr=shr, shrs=shrs,
    maxs=maxs, mins=mins, prelu=prelu,
    addc=addc, subc=subc, addsc=addsc, subsc=subsc,
    lrelu=lrelu, sel=sel, sels=sels,
)

tensor = _types.SimpleNamespace(
    # Overlapping (tensor versions)
    add=_tensor_add, sub=_tensor_sub, mul=_tensor_mul, div=_tensor_div,
    exp=_tensor_exp, cast=_tensor_cast, matmul=_tensor_matmul,
    row_max=_tensor_row_max, row_sum=_tensor_row_sum, maximum=_tensor_maximum,
    reshape=_tensor_reshape, transpose=_tensor_transpose, view=_tensor_view,
    getval=_tensor_getval, setval=_tensor_setval,
    # Non-overlapping
    create_tensor=create_tensor, read=read, dim=dim,
    mul_scalar=mul_scalar, add_scalar=add_scalar,
    sub_scalar=sub_scalar, div_scalar=div_scalar,
    assemble=assemble,
)


# ###########################################################################
#  __all__
# ###########################################################################

__all__ = [
    # Namespace objects
    "block",
    "tensor",
    # Unified dispatch
    "add", "sub", "mul", "div", "maximum", "exp", "cast",
    "reshape", "transpose", "view", "matmul",
    "row_max", "row_sum",
    "getval", "setval",
    # Block-only (promoted)
    "make_tile", "load", "store", "l0c_store", "move", "vec_move",
    "full", "fillpad",
    "get_block_idx", "get_subblock_idx", "get_block_num", "index_cast",
    "adds", "subs", "muls", "divs",
    "neg", "sqrt", "rsqrt", "recip", "log", "abs", "relu", "not_",
    "matmul_acc", "matmul_bias",
    "gemv", "gemv_acc", "gemv_bias",
    "row_min", "minimum",
    "row_expand", "row_expand_sub", "row_expand_div", "row_expand_mul", "row_expand_add",
    "col_expand", "col_expand_mul", "col_expand_div", "col_expand_sub",
    "expands", "cmp", "cmps",
    "sum", "max", "min",
    "rem", "rems",
    "and_", "ands", "or_", "ors", "xor", "xors",
    "shl", "shls", "shr", "shrs",
    "maxs", "mins", "prelu",
    "addc", "subc", "addsc", "subsc",
    "lrelu", "sel", "sels",
    # Tensor-only (promoted)
    "create_tensor", "read", "dim", "assemble",
    "mul_scalar", "add_scalar", "sub_scalar", "div_scalar",
]
