# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for tuple literal and subscript syntax in parser."""

import pypto.language as pl
import pytest
from pypto import ir


class TestTupleLiteralParsing:
    """Tests for parsing tuple literals (x, y, z)."""

    def test_parse_empty_tuple(self):
        """Test parsing empty tuple literal."""

        @pl.function
        def func():
            _ = ()

        # Verify function was created
        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_tuple_with_two_elements(self):
        """Test parsing tuple with two elements."""

        @pl.function
        def func(x: pl.Tensor[[10], pl.FP32], y: pl.Scalar[pl.INT64]):
            _ = (x, y)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_tuple_with_constants(self):
        """Test parsing tuple with constant values."""

        @pl.function
        def func():
            _ = (1, 2, 3)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_nested_tuple(self):
        """Test parsing nested tuples."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            inner = (x, x)
            _ = (inner, x)

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_singleton_tuple(self):
        """Test parsing single element tuple."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64]):
            _ = (x,)

        assert func is not None
        assert isinstance(func, ir.Function)


class TestTupleSubscriptParsing:
    """Tests for parsing tuple subscript access tuple[0]."""

    def test_parse_simple_subscript(self):
        """Test parsing simple tuple subscript - need to create tuple first."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _ = my_tuple[0]
            _ = my_tuple[1]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_parse_nested_subscript(self):
        """Test parsing nested tuple subscript."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            inner = (x, x)
            nested = (inner, y)
            _ = nested[0]
            _ = nested[0][1]

        assert func is not None
        assert isinstance(func, ir.Function)


class TestTupleRoundTrip:
    """Tests for creating and accessing tuples."""

    def test_create_and_access_tuple(self):
        """Test creating tuple and accessing elements."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32]):
            my_tuple = (x, y)
            _ = my_tuple[0]
            _ = my_tuple[1]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_tuple_in_operations(self):
        """Test using tuple elements in operations."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64]):
            my_tuple = (x, y)
            # Access tuple elements
            first = my_tuple[0]
            second = my_tuple[1]
            # Store them for verification
            _ = first
            _ = second

        assert func is not None
        assert isinstance(func, ir.Function)


class TestTupleVariableIndexParsing:
    """Tests for variable index access on tuples (lowered to if-else chain)."""

    def test_variable_index_homogeneous_tuple(self):
        """Variable index on a homogeneous tuple generates valid IR."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64], idx: pl.Scalar[pl.INT64]):
            my_tuple = (x, y)
            _ = my_tuple[idx]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_variable_index_three_elements(self):
        """Variable index on a 3-element homogeneous tuple."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            c: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            my_tuple = (a, b, c)
            _ = my_tuple[idx]

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_variable_index_result_used_in_expression(self):
        """Variable index result can be used in subsequent expressions."""

        @pl.function
        def func(
            x: pl.Scalar[pl.INT64],
            y: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            my_tuple = (x, y)
            val = my_tuple[idx]
            _ = val + x

        assert func is not None
        assert isinstance(func, ir.Function)

    def test_variable_index_heterogeneous_tuple_raises(self):
        """Variable index on heterogeneous tuple raises an error."""
        with pytest.raises(Exception):

            @pl.function
            def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.FP32], idx: pl.Scalar[pl.INT64]):
                my_tuple = (x, y)
                _ = my_tuple[idx]

    def test_variable_index_ir_contains_if_stmt(self):
        """Variable index generates IfStmt nodes in the IR."""

        @pl.function
        def func(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64], idx: pl.Scalar[pl.INT64]):
            my_tuple = (x, y)
            _ = my_tuple[idx]

        assert func is not None
        body_stmts = func.body.stmts
        if_stmts = [s for s in body_stmts if isinstance(s, ir.IfStmt)]
        assert len(if_stmts) >= 1


def _top_level_if_stmts(func) -> list:
    """Return IfStmt nodes at the top level of a function body (not nested)."""
    body = func.body
    stmts = body.stmts if hasattr(body, "stmts") else [body]
    return [s for s in stmts if isinstance(s, ir.IfStmt)]


class TestTupleSelectCache:
    """Tests for the _tuple_select_cache that deduplicates buf[idx] expansions.

    When the same (tuple_var_name, index_var_name) pair is accessed more than
    once, the parser must reuse the already-emitted phi Var instead of building
    a redundant scf.if chain.
    """

    def test_duplicate_access_produces_single_if_stmt(self):
        """buf[idx] accessed twice → only one IfStmt at the function top level."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            buf = (a, b)
            v1 = buf[idx]
            v2 = buf[idx]   # cache hit — no new IfStmt
            _ = v1 + v2

        assert isinstance(func, ir.Function)
        if_stmts = _top_level_if_stmts(func)
        # With cache: only 1 IfStmt despite 2 buf[idx] accesses
        assert len(if_stmts) == 1

    def test_triple_access_still_single_if_stmt(self):
        """buf[idx] accessed three times → still only one IfStmt."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            buf = (a, b)
            v1 = buf[idx]
            v2 = buf[idx]   # cache hit
            v3 = buf[idx]   # cache hit again
            _ = v1 + v2 + v3

        assert isinstance(func, ir.Function)
        if_stmts = _top_level_if_stmts(func)
        assert len(if_stmts) == 1

    def test_different_tuples_same_idx_separate_chains(self):
        """buf1[idx] and buf2[idx] have different tuple names → two IfStmts."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            buf1 = (a, b)
            buf2 = (b, a)
            v1 = buf1[idx]
            v2 = buf2[idx]  # different tuple var name → new chain
            _ = v1 + v2

        assert isinstance(func, ir.Function)
        if_stmts = _top_level_if_stmts(func)
        assert len(if_stmts) == 2

    def test_same_tuple_different_idx_separate_chains(self):
        """buf[idx1] and buf[idx2] have different index vars → two IfStmts."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            idx0: pl.Scalar[pl.INT64],
            idx1: pl.Scalar[pl.INT64],
        ):
            buf = (a, b)
            v1 = buf[idx0]
            v2 = buf[idx1]  # different index var → new chain
            _ = v1 + v2

        assert isinstance(func, ir.Function)
        if_stmts = _top_level_if_stmts(func)
        assert len(if_stmts) == 2

    def test_cached_result_is_same_var(self):
        """Both accesses must resolve to the same phi Var name in the IR."""

        @pl.function
        def func(
            a: pl.Scalar[pl.INT64],
            b: pl.Scalar[pl.INT64],
            idx: pl.Scalar[pl.INT64],
        ):
            buf = (a, b)
            v1 = buf[idx]
            v2 = buf[idx]   # cache hit

        assert isinstance(func, ir.Function)
        body = func.body
        stmts = body.stmts if hasattr(body, "stmts") else [body]
        assign_stmts = [s for s in stmts if isinstance(s, ir.AssignStmt)]
        # Find v1 and v2 assignments (skip the buf = MakeTuple assignment)
        tile_assigns = [s for s in assign_stmts if s.var.name in ("v1", "v2")]
        assert len(tile_assigns) == 2
        # Both v1 and v2 must be assigned the same phi Var
        v1_rhs = tile_assigns[0].value
        v2_rhs = tile_assigns[1].value
        assert isinstance(v1_rhs, ir.Var)
        assert isinstance(v2_rhs, ir.Var)
        assert v1_rhs.name == v2_rhs.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
