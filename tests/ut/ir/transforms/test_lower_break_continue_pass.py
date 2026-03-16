# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LowerBreakContinue pass.

The pass transforms break/continue IR statements into structured control flow
(nested scf.if) so that code generation only sees ForStmt and WhileStmt with
iter_args and YieldStmt — no BreakStmt or ContinueStmt.

Coverage:
  - Continue in for loop (no / with iter_args)
  - Break in for loop (no / with iter_args)
  - Continue in while loop (no iter_args)
  - Break in while loop (no iter_args)
  - No-op: loops with no break/continue are returned unchanged
"""

import pytest
from pypto import DataType, ir, passes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPAN = ir.Span.unknown()


def _int(n: int) -> ir.ConstInt:
    return ir.ConstInt(n, DataType.INT64, SPAN)


def _bool(v: bool) -> ir.ConstBool:
    return ir.ConstBool(v, SPAN)


def _make_program(body: ir.Stmt, name: str = "main") -> ir.Program:
    func = ir.Function(name, [], [], body, SPAN)
    return ir.Program([func], "test", SPAN)


def _run_pass(prog: ir.Program) -> ir.Stmt:
    """Run lower_break_continue and return the transformed function body."""
    after = passes.lower_break_continue()(prog)
    return list(after.functions.values())[0].body


def _contains_break(stmt: ir.Stmt) -> bool:
    """Return True if stmt subtree contains any BreakStmt."""
    if isinstance(stmt, ir.BreakStmt):
        return True
    for attr in ("stmts", "then_body", "else_body", "body"):
        child = getattr(stmt, attr, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            if any(_contains_break(s) for s in child):
                return True
        elif isinstance(child, ir.Stmt):
            if _contains_break(child):
                return True
    return False


def _contains_continue(stmt: ir.Stmt) -> bool:
    """Return True if stmt subtree contains any ContinueStmt."""
    if isinstance(stmt, ir.ContinueStmt):
        return True
    for attr in ("stmts", "then_body", "else_body", "body"):
        child = getattr(stmt, attr, None)
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            if any(_contains_continue(s) for s in child):
                return True
        elif isinstance(child, ir.Stmt):
            if _contains_continue(child):
                return True
    return False


# ---------------------------------------------------------------------------
# No-op: loops without break/continue are returned unchanged
# ---------------------------------------------------------------------------


def test_noop_for_no_break_continue():
    """For loop without break/continue is returned unchanged (same object)."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    assign = ir.AssignStmt(
        ir.Var("x", ir.ScalarType(DataType.INT64), SPAN),
        ir.Add(i, _int(1), DataType.INT64, SPAN),
        SPAN,
    )
    loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [], assign, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_break(result)
    assert not _contains_continue(result)


def test_noop_while_no_break_continue():
    """While loop without break/continue is returned unchanged."""
    cond = ir.Lt(_int(0), _int(10), DataType.BOOL, SPAN)
    assign = ir.AssignStmt(
        ir.Var("x", ir.ScalarType(DataType.INT64), SPAN),
        _int(1),
        SPAN,
    )
    loop = ir.WhileStmt(cond, [], assign, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.WhileStmt)
    assert not _contains_break(result)
    assert not _contains_continue(result)


# ---------------------------------------------------------------------------
# Continue in for loop
# ---------------------------------------------------------------------------


def test_continue_in_for_loop_no_iter_args():
    """if (cond) { continue }; rest  →  if (!cond) { rest }."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    cond = ir.Lt(i, _int(5), DataType.BOOL, SPAN)
    cont = ir.ContinueStmt(SPAN)
    if_cont = ir.IfStmt(cond, cont, None, [], SPAN)
    assign = ir.AssignStmt(
        ir.Var("x", ir.ScalarType(DataType.INT64), SPAN),
        ir.Add(i, _int(1), DataType.INT64, SPAN),
        SPAN,
    )
    body = ir.SeqStmts([if_cont, assign], SPAN)
    loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [], body, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_continue(result)

    # Body must be: if (i >= 5) { assign }  (Lt negated to Ge by NegateCondition)
    guard = result.body
    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.Ge)
    assert isinstance(guard.then_body, ir.AssignStmt)


def test_continue_in_for_loop_with_iter_args():
    """Continue in for loop with iter_args: else branch yields current iter_arg values."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    x_ia = ir.IterArg("x", ir.ScalarType(DataType.INT64), _int(0), SPAN)
    x_ret = ir.Var("x_final", ir.ScalarType(DataType.INT64), SPAN)

    cond = ir.Lt(i, _int(5), DataType.BOOL, SPAN)
    cont = ir.ContinueStmt(SPAN)
    if_cont = ir.IfStmt(cond, cont, None, [], SPAN)
    new_x = ir.Add(x_ia, i, DataType.INT64, SPAN)
    yield_stmt = ir.YieldStmt([new_x], SPAN)
    body = ir.SeqStmts([if_cont, yield_stmt], SPAN)

    loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [x_ia], body, [x_ret], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_continue(result)

    # Transformed body: if (i >= 5) { yield(x+i) } else { yield(x_ia) }
    # (Lt condition negated to Ge by NegateCondition)
    guard = result.body
    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.Ge)
    assert guard.else_body is not None
    assert isinstance(guard.else_body, ir.YieldStmt)


# ---------------------------------------------------------------------------
# Break in for loop
# ---------------------------------------------------------------------------


def test_break_in_for_loop_no_iter_args():
    """if (cond) { break } → adds _can_continue iter_arg and guards the body."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    cond = ir.Lt(i, _int(5), DataType.BOOL, SPAN)
    brk = ir.BreakStmt(SPAN)
    if_brk = ir.IfStmt(cond, brk, None, [], SPAN)
    loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [], if_brk, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_break(result)

    # Must have added _can_continue iter_arg
    assert len(result.iter_args) == 1
    assert result.iter_args[0].name == "_can_continue"
    assert len(result.return_vars) == 1

    # Body must be SeqStmts([guard_if, trailing_yield_to_for])
    seq = result.body
    assert isinstance(seq, ir.SeqStmts)
    assert len(seq.stmts) == 2

    # First stmt: guard if (_can_continue) { ... } else { yield(_can_continue) }
    # No NOT needed — _can_continue is already false when break was taken
    guard = seq.stmts[0]
    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.IterArg)
    assert guard.else_body is not None
    assert isinstance(guard.else_body, ir.YieldStmt)

    # Guard then-branch contains the break-site value-producing IfStmt
    then = guard.then_body
    assert isinstance(then, ir.SeqStmts)
    break_site_if = then.stmts[0]
    assert isinstance(break_site_if, ir.IfStmt)
    # Break path yields False (cannot continue)
    assert isinstance(break_site_if.then_body, ir.YieldStmt)
    assert len(break_site_if.then_body.value) == 1
    assert isinstance(break_site_if.then_body.value[0], ir.ConstBool)
    # Propagation yield at end of then-branch references the break-site result var
    prop_yield = then.stmts[-1]
    assert isinstance(prop_yield, ir.YieldStmt)

    # Trailing yield feeds the for loop's scf.yield
    for_yield = seq.stmts[-1]
    assert isinstance(for_yield, ir.YieldStmt)


def test_break_in_for_loop_with_iter_args():
    """Break in for loop with iter_args: _can_continue prepended to existing iter_args."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    x_ia = ir.IterArg("x", ir.ScalarType(DataType.INT64), _int(0), SPAN)
    x_ret = ir.Var("x_final", ir.ScalarType(DataType.INT64), SPAN)

    cond = ir.Lt(i, _int(5), DataType.BOOL, SPAN)
    brk = ir.BreakStmt(SPAN)
    if_brk = ir.IfStmt(cond, brk, None, [], SPAN)
    yield_stmt = ir.YieldStmt([ir.Add(x_ia, i, DataType.INT64, SPAN)], SPAN)
    body = ir.SeqStmts([if_brk, yield_stmt], SPAN)

    loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [x_ia], body, [x_ret], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_break(result)

    # _can_continue prepended to the original iter_args
    assert len(result.iter_args) == 2
    assert result.iter_args[0].name == "_can_continue"
    assert len(result.return_vars) == 2

    # Body is SeqStmts([guard_if, trailing_yield])
    seq = result.body
    assert isinstance(seq, ir.SeqStmts)
    guard = seq.stmts[0]
    assert isinstance(guard, ir.IfStmt)
    # Guard condition is _can_continue directly — no NOT needed
    assert isinstance(guard.condition, ir.IterArg)
    # Guard result vars: [_guard_cont, _guard_ia0] → 2 total (one per new iter_arg)
    assert len(guard.return_vars) == 2


# ---------------------------------------------------------------------------
# Continue in while loop
# ---------------------------------------------------------------------------


def test_continue_in_while_loop():
    """Continue in while loop: remaining stmts wrapped in if(!cond)."""
    cond_w = ir.Lt(_int(0), _int(10), DataType.BOOL, SPAN)
    iter_cond = ir.Lt(
        ir.Var("dummy", ir.ScalarType(DataType.INT64), SPAN),
        _int(3),
        DataType.BOOL,
        SPAN,
    )
    cont = ir.ContinueStmt(SPAN)
    if_cont = ir.IfStmt(iter_cond, cont, None, [], SPAN)
    assign = ir.AssignStmt(
        ir.Var("x", ir.ScalarType(DataType.INT64), SPAN),
        _int(1),
        SPAN,
    )
    body = ir.SeqStmts([if_cont, assign], SPAN)
    loop = ir.WhileStmt(cond_w, [], body, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.WhileStmt)
    assert not _contains_continue(result)

    # Body must be: if (dummy >= 3) { assign }  (Lt negated to Ge by NegateCondition)
    guard = result.body
    assert isinstance(guard, ir.IfStmt)
    assert isinstance(guard.condition, ir.Ge)


# ---------------------------------------------------------------------------
# Break in while loop
# ---------------------------------------------------------------------------


def test_break_in_while_loop():
    """Break in while loop: _can_continue is sole before-region condition; original
    condition is checked via scf.if at the start of the do-region (avoids And)."""
    cond_w = ir.Lt(_int(0), _int(10), DataType.BOOL, SPAN)
    brk = ir.BreakStmt(SPAN)
    iter_cond = ir.Lt(_int(5), _int(3), DataType.BOOL, SPAN)
    if_brk = ir.IfStmt(iter_cond, brk, None, [], SPAN)
    loop = ir.WhileStmt(cond_w, [], if_brk, [], SPAN)
    prog = _make_program(loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.WhileStmt)
    assert not _contains_break(result)

    # _can_continue added to iter_args
    assert len(result.iter_args) == 1
    assert result.iter_args[0].name == "_can_continue"
    assert len(result.return_vars) == 1

    # Condition is just _can_continue — no And needed
    assert isinstance(result.condition, ir.IterArg)
    assert result.condition.name == "_can_continue"

    # Body is SeqStmts([outer_if, trailing_yield])
    seq = result.body
    assert isinstance(seq, ir.SeqStmts)
    assert len(seq.stmts) == 2

    # outer_if checks the original condition: if (orig_cond) { then } else { yield(false,...) }
    outer_if = seq.stmts[0]
    assert isinstance(outer_if, ir.IfStmt)
    assert isinstance(outer_if.condition, ir.Lt)   # original cond_w preserved
    assert outer_if.else_body is not None
    # else-branch is YieldStmt([false]) — signals "cond false, stop"
    assert isinstance(outer_if.else_body, ir.YieldStmt)
    assert len(outer_if.else_body.value) == 1
    assert isinstance(outer_if.else_body.value[0], ir.ConstBool)
    assert outer_if.else_body.value[0].value is False

    # trailing yield feeds scf.while's scf.yield
    trailing = seq.stmts[1]
    assert isinstance(trailing, ir.YieldStmt)


# ---------------------------------------------------------------------------
# Nested loops: inner break/continue doesn't affect outer loop
# ---------------------------------------------------------------------------


def test_break_in_nested_for_loop():
    """Break in inner loop only transforms the inner loop, not the outer."""
    i = ir.Var("i", ir.ScalarType(DataType.INT64), SPAN)
    j = ir.Var("j", ir.ScalarType(DataType.INT64), SPAN)
    cond = ir.Lt(j, _int(3), DataType.BOOL, SPAN)
    brk = ir.BreakStmt(SPAN)
    if_brk = ir.IfStmt(cond, brk, None, [], SPAN)
    inner_loop = ir.ForStmt(j, _int(0), _int(5), _int(1), [], if_brk, [], SPAN)
    outer_loop = ir.ForStmt(i, _int(0), _int(10), _int(1), [], inner_loop, [], SPAN)
    prog = _make_program(outer_loop)

    result = _run_pass(prog)
    assert isinstance(result, ir.ForStmt)
    assert not _contains_break(result)

    # Outer loop should NOT have _should_break iter_arg
    assert len(result.iter_args) == 0

    # Inner loop SHOULD have _can_continue iter_arg
    inner = result.body
    assert isinstance(inner, ir.ForStmt)
    assert len(inner.iter_args) == 1
    assert inner.iter_args[0].name == "_can_continue"

    # Inner loop body is SeqStmts([guard_if, trailing_yield])
    inner_seq = inner.body
    assert isinstance(inner_seq, ir.SeqStmts)
    assert isinstance(inner_seq.stmts[0], ir.IfStmt)  # guard_if


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
