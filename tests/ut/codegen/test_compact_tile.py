# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the CompactMode field on TileView/TileType.

Covers:
- CompactMode enum values accessible from Python
- TileType.compact field in the Python DSL descriptor
- compact kwarg propagation through create_op_call → C++ DeduceBlockCreateTileType
- compact propagation through the @pl.function AST parser (TileType special-case path)
- compact field emitted (or omitted) in PTOAS MLIR tile_buf type strings
"""

import pypto.language as pl
import pypto.language.manual as plm
import pytest
from pypto import DataType, backend
from pypto.backend import BackendType
from pypto.ir.op import block_ops as ir_block_ops
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes as _passes
from pypto.pypto_core.codegen import PTOCodegen
from pypto.pypto_core.ir import CompactMode, MemorySpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_M = pl.DynVar("M")
_K = pl.DynVar("K")


def _compile_kernel_to_mlir(func: _ir.Function) -> str:
    """Run minimal passes and PTOCodegen on a single @pl.function."""
    prog = _ir.Program([func], "compact_test", _ir.Span.unknown())
    pipeline = _passes.PassPipeline()
    pipeline.add_pass(_passes.lower_break_continue())
    pipeline.add_pass(_passes.convert_to_ssa())
    pipeline.add_pass(_passes.init_mem_ref())
    with _passes.PassContext([]):
        result = pipeline.run(prog)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    cg = PTOCodegen()
    return cg.generate(result)


def _alloc_tile_lines(mlir: str) -> list[str]:
    """Return all pto.alloc_tile lines from an MLIR string."""
    return [line.strip() for line in mlir.splitlines() if "alloc_tile" in line]


# ---------------------------------------------------------------------------
# CompactMode enum
# ---------------------------------------------------------------------------

class TestCompactModeEnum:
    """Tests for the CompactMode enum exposed via nanobind."""

    def test_null_value_exists(self):
        assert hasattr(CompactMode, "null")

    def test_normal_value_exists(self):
        assert hasattr(CompactMode, "normal")

    def test_row_plus_one_value_exists(self):
        assert hasattr(CompactMode, "row_plus_one")

    def test_null_is_default(self):
        tv = _ir.TileView()
        assert tv.compact == CompactMode.null

    def test_compact_readable_and_writable(self):
        tv = _ir.TileView()
        tv.compact = CompactMode.normal
        assert tv.compact == CompactMode.normal
        tv.compact = CompactMode.row_plus_one
        assert tv.compact == CompactMode.row_plus_one
        tv.compact = CompactMode.null
        assert tv.compact == CompactMode.null


# ---------------------------------------------------------------------------
# Python-layer TileType descriptor
# ---------------------------------------------------------------------------

class TestTileTypeCompactField:
    """Tests for plm.TileType.compact (Python DSL descriptor)."""

    def test_compact_defaults_to_none(self):
        tt = plm.TileType(shape=[128, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
        assert tt.compact is None

    def test_compact_set_to_normal(self):
        tt = plm.TileType(shape=[128, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, compact=1)
        assert tt.compact == 1

    def test_compact_set_to_row_plus_one(self):
        tt = plm.TileType(shape=[128, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec, compact=2)
        assert tt.compact == 2


# ---------------------------------------------------------------------------
# IR level: create_op_call propagates compact into TileType.tile_view
# ---------------------------------------------------------------------------

class TestCompactInIR:
    """Tests that compact=N is stored in TileType.tile_view.compact after create_op_call."""

    def test_compact_null_by_default(self):
        call = ir_block_ops.make_tile(
            shape=[128, 128],
            dtype=DataType.FP16,
            target_memory=MemorySpace.Left,
            addr=0,
            size=32768,
            blayout=1,
            slayout=1,
        )
        t = call.type
        assert t.tile_view is not None
        assert t.tile_view.compact == CompactMode.null

    def test_compact_normal_stored_in_tileview(self):
        call = ir_block_ops.make_tile(
            shape=[128, 128],
            dtype=DataType.FP16,
            target_memory=MemorySpace.Left,
            addr=0,
            size=32768,
            blayout=1,
            slayout=1,
            compact=1,
        )
        t = call.type
        assert t.tile_view is not None
        assert t.tile_view.compact == CompactMode.normal

    def test_compact_row_plus_one_stored_in_tileview(self):
        call = ir_block_ops.make_tile(
            shape=[128, 128],
            dtype=DataType.FP16,
            target_memory=MemorySpace.Right,
            addr=0,
            size=32768,
            blayout=1,
            slayout=2,
            compact=2,
        )
        t = call.type
        assert t.tile_view is not None
        assert t.tile_view.compact == CompactMode.row_plus_one


# ---------------------------------------------------------------------------
# AST parser: compact propagated through @pl.function TileType special-case
# ---------------------------------------------------------------------------

class TestCompactASTParser:
    """Tests that the @pl.function AST parser forwards compact from TileType to the IR."""

    def test_compact_propagated_to_ir_normal(self):
        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            tile_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Left,
                blayout=1,
                slayout=1,
                compact=1,
            )
            tile_a = plm.make_tile(tile_type, addr=0x00000, size=32768)

        # Navigate to the AssignStmt for tile_a and check its type
        body = func.body
        stmt = body.stmts[0] if hasattr(body, "stmts") else body
        t = stmt.var.type
        assert t.tile_view is not None
        assert t.tile_view.compact == CompactMode.normal

    def test_compact_absent_when_not_set(self):
        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            tile_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Left,
                blayout=1,
                slayout=1,
            )
            tile_a = plm.make_tile(tile_type, addr=0x00000, size=32768)

        body = func.body
        stmt = body.stmts[0] if hasattr(body, "stmts") else body
        t = stmt.var.type
        assert t.tile_view is not None
        assert t.tile_view.compact == CompactMode.null


# ---------------------------------------------------------------------------
# Codegen: compact emitted / omitted in PTOAS tile_buf type strings
# ---------------------------------------------------------------------------

class TestCompactCodegen:
    """Tests that compact is correctly emitted in the PTOAS MLIR tile_buf type string."""

    def test_compact_normal_emitted_as_compact_1(self):
        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            tile_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Left,
                blayout=1,
                slayout=1,
                valid_shape=[-1, -1],
                compact=1,
            )
            tile_a = plm.make_tile(tile_type, addr=0x00000, size=32768)

        mlir = _compile_kernel_to_mlir(func)
        alloc_lines = _alloc_tile_lines(mlir)
        assert len(alloc_lines) >= 1
        assert any("compact=1" in line for line in alloc_lines)

    def test_compact_row_plus_one_emitted_as_compact_2(self):
        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            tile_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Right,
                blayout=1,
                slayout=2,
                valid_shape=[-1, -1],
                compact=2,
            )
            tile_b = plm.make_tile(tile_type, addr=0x08000, size=32768)

        mlir = _compile_kernel_to_mlir(func)
        alloc_lines = _alloc_tile_lines(mlir)
        assert len(alloc_lines) >= 1
        assert any("compact=2" in line for line in alloc_lines)

    def test_compact_null_not_emitted(self):
        """When compact is not set (null), compact= must NOT appear in tile_buf."""

        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            tile_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            )
            tile_c = plm.make_tile(tile_type, addr=0x10000, size=32768)

        mlir = _compile_kernel_to_mlir(func)
        alloc_lines = _alloc_tile_lines(mlir)
        assert len(alloc_lines) >= 1
        assert all("compact=" not in line for line in alloc_lines)

    def test_mixed_compact_and_null_tiles(self):
        """Left tile with compact=1, Vec tile without compact."""

        @pl.function
        def func(a: pl.Tensor[[_M, _K], pl.FP16]) -> pl.Tensor[[_M, _K], pl.FP16]:
            left_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Left,
                blayout=1,
                slayout=1,
                valid_shape=[-1, -1],
                compact=1,
            )
            tile_left = plm.make_tile(left_type, addr=0x00000, size=32768)
            vec_type = plm.TileType(
                shape=[128, 128],
                dtype=pl.FP16,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[-1, -1],
            )
            tile_vec = plm.make_tile(vec_type, addr=0x08000, size=32768)

        mlir = _compile_kernel_to_mlir(func)
        alloc_lines = _alloc_tile_lines(mlir)
        assert len(alloc_lines) == 2
        compact_lines = [l for l in alloc_lines if "compact=" in l]
        no_compact_lines = [l for l in alloc_lines if "compact=" not in l]
        assert len(compact_lines) == 1
        assert len(no_compact_lines) == 1
        assert "compact=1" in compact_lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
