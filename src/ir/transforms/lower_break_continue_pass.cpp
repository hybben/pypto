/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// ============================================================================
// Helpers
// ============================================================================

/// Check if a statement subtree contains any BreakStmt nodes.
class ContainsBreakVisitor : public IRVisitor {
 public:
  bool found = false;

 protected:
  void VisitStmt_(const BreakStmtPtr& /*op*/) override { found = true; }
  // Don't recurse into nested loops — their breaks don't affect the outer loop
  void VisitStmt_(const ForStmtPtr& /*op*/) override {}
  void VisitStmt_(const WhileStmtPtr& /*op*/) override {}
};

/// Check if a statement subtree contains any ContinueStmt nodes.
class ContainsContinueVisitor : public IRVisitor {
 public:
  bool found = false;

 protected:
  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { found = true; }
  // Don't recurse into nested loops
  void VisitStmt_(const ForStmtPtr& /*op*/) override {}
  void VisitStmt_(const WhileStmtPtr& /*op*/) override {}
};

bool ContainsBreak(const StmtPtr& stmt) {
  ContainsBreakVisitor v;
  v.VisitStmt(stmt);
  return v.found;
}

bool ContainsContinue(const StmtPtr& stmt) {
  ContainsContinueVisitor v;
  v.VisitStmt(stmt);
  return v.found;
}

/// Negate a boolean condition, folding into the inverted comparison when possible.
///
/// Comparison inversion avoids emitting an extra NOT op in codegen:
///   NOT (a < b)  →  a >= b       (arith.cmpi slt → arith.cmpi sge)
///   NOT (a <= b) →  a > b
///   NOT (a > b)  →  a <= b
///   NOT (a >= b) →  a < b
///   NOT (a == b) →  a != b
///   NOT (a != b) →  a == b
///   NOT (NOT x)  →  x            (double-negation elimination)
///
/// Falls back to Not(cond) for non-comparison operands (e.g., boolean variables).
ExprPtr NegateCondition(const ExprPtr& cond, const Span& span) {
  if (auto e = As<Lt>(cond)) return std::make_shared<Ge>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Le>(cond)) return std::make_shared<Gt>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Gt>(cond)) return std::make_shared<Le>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Ge>(cond)) return std::make_shared<Lt>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Eq>(cond)) return std::make_shared<Ne>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Ne>(cond)) return std::make_shared<Eq>(e->left_, e->right_, DataType::BOOL, span);
  if (auto e = As<Not>(cond)) return e->operand_;  // NOT (NOT x) → x
  return std::make_shared<Not>(cond, DataType::BOOL, span);
}

/// Collect statements from a SeqStmts or wrap a single statement in a vector.
std::vector<StmtPtr> FlattenToStmtList(const StmtPtr& stmt) {
  if (auto seq = As<SeqStmts>(stmt)) {
    return seq->stmts_;
  }
  return {stmt};
}

/// Create a SeqStmts or return the single statement directly.
StmtPtr MakeSeqOrSingle(std::vector<StmtPtr> stmts, const Span& span) {
  if (stmts.size() == 1) {
    return stmts[0];
  }
  return std::make_shared<SeqStmts>(std::move(stmts), span);
}

// ============================================================================
// Continue Lowering
// ============================================================================

/// Lower continue statements in a list of statements.
///
/// Pattern:
///   stmt_before
///   if (cond) { continue }
///   stmt_after_1
///   stmt_after_2
///
/// Becomes:
///   stmt_before
///   if (!cond) {
///     stmt_after_1
///     stmt_after_2
///   }
///
/// For loops with iter_args, the else branch yields current iter_arg values.
std::vector<StmtPtr> LowerContinueInStmtList(const std::vector<StmtPtr>& stmts,
                                              const std::vector<IterArgPtr>& iter_args);

/// Recursively lower continue in a single statement.
StmtPtr LowerContinueInStmt(const StmtPtr& stmt, const std::vector<IterArgPtr>& iter_args) {
  // IfStmt whose then-body ends with continue
  if (auto if_stmt = As<IfStmt>(stmt)) {
    auto then_stmts = FlattenToStmtList(if_stmt->then_body_);
    bool then_ends_with_continue =
        !then_stmts.empty() && As<ContinueStmt>(then_stmts.back()) != nullptr;

    if (then_ends_with_continue && !if_stmt->else_body_.has_value()) {
      if (then_stmts.size() == 1) {
        // `if (cond) { continue }` — handled at the SeqStmts level when there are
        // remaining statements; when called standalone just drop the continue.
        return stmt;
      }
      // `if (cond) { pre_stmts...; continue }` without remaining — strip the trailing
      // continue and return the modified if.  When there ARE remaining statements the
      // SeqStmts-level handler intercepts this case before calling us.
      std::vector<StmtPtr> pre_stmts(then_stmts.begin(), then_stmts.end() - 1);
      auto lowered_pre = LowerContinueInStmtList(pre_stmts, iter_args);
      StmtPtr new_then_body = MakeSeqOrSingle(std::move(lowered_pre), if_stmt->span_);
      return std::make_shared<IfStmt>(if_stmt->condition_, std::move(new_then_body), std::nullopt,
                                      if_stmt->return_vars_, if_stmt->span_);
    }

    // Check if else-body has a continue
    if (if_stmt->else_body_.has_value()) {
      auto else_stmts = FlattenToStmtList(*if_stmt->else_body_);
      bool else_is_continue = (else_stmts.size() == 1 && As<ContinueStmt>(else_stmts[0]) != nullptr);

      if (else_is_continue) {
        // `if (cond) { body } else { continue }` → `if (cond) { body }`
        // The remaining statements after this if will need to be guarded by cond
        return stmt;  // Handled at SeqStmts level
      }
    }

    // Recurse into branches
    StmtPtr new_then = LowerContinueInStmt(if_stmt->then_body_, iter_args);
    std::optional<StmtPtr> new_else;
    if (if_stmt->else_body_.has_value()) {
      new_else = LowerContinueInStmt(*if_stmt->else_body_, iter_args);
    }
    if (new_then != if_stmt->then_body_ || new_else != if_stmt->else_body_) {
      return std::make_shared<IfStmt>(if_stmt->condition_, std::move(new_then), std::move(new_else),
                                      if_stmt->return_vars_, if_stmt->span_);
    }
    return stmt;
  }

  // SeqStmts — look for continue patterns in the list
  if (auto seq = As<SeqStmts>(stmt)) {
    auto lowered = LowerContinueInStmtList(seq->stmts_, iter_args);
    if (lowered.size() == 1) return lowered[0];
    return std::make_shared<SeqStmts>(std::move(lowered), seq->span_);
  }

  return stmt;
}

std::vector<StmtPtr> LowerContinueInStmtList(const std::vector<StmtPtr>& stmts,
                                              const std::vector<IterArgPtr>& iter_args) {
  std::vector<StmtPtr> result;

  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Check for `if (cond) { [pre_stmts...;] continue }` pattern
    if (auto if_stmt = As<IfStmt>(stmt)) {
      auto then_stmts = FlattenToStmtList(if_stmt->then_body_);
      bool then_ends_with_continue =
          !then_stmts.empty() && As<ContinueStmt>(then_stmts.back()) != nullptr;

      if (then_ends_with_continue && !if_stmt->else_body_.has_value()) {
        // Collect statements before the trailing continue (may be empty).
        std::vector<StmtPtr> pre_stmts(then_stmts.begin(), then_stmts.end() - 1);

        std::vector<StmtPtr> remaining(stmts.begin() + i + 1, stmts.end());
        auto lowered_remaining = LowerContinueInStmtList(remaining, iter_args);

        if (!pre_stmts.empty()) {
          // Pattern: if (cond) { pre_stmts; continue }; remaining...
          //
          // ── Emit as a single if-else, NOT two separate ifs ──
          //
          //   if (cond) { pre_stmts } else { remaining }
          //
          // This is critical for correctness: both branches evaluate the SAME
          // original condition value.  If we emitted two separate ifs:
          //   if (cond)  { pre_stmts }       ← may update variables used in cond
          //   if (!cond) { remaining }        ← !cond now tests the *updated* value
          // ConvertToSSA would then chain variable versions across the two ifs,
          // causing the second condition to test a post-update value and producing
          // wrong control flow (e.g. tadd executing when loop==1 despite a continue).
          auto lowered_pre = LowerContinueInStmtList(pre_stmts, iter_args);
          StmtPtr then_body = MakeSeqOrSingle(std::move(lowered_pre), if_stmt->span_);

          if (lowered_remaining.empty()) {
            // No remaining stmts — just emit  if (cond) { pre_stmts }
            if (!iter_args.empty()) {
              // SSA loop: else branch must yield current iter_arg values (continue path)
              std::vector<ExprPtr> yield_values;
              for (const auto& ia : iter_args) yield_values.push_back(ia);
              StmtPtr else_body = std::make_shared<YieldStmt>(std::move(yield_values), if_stmt->span_);
              result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, std::move(then_body),
                                                        std::make_optional(std::move(else_body)),
                                                        std::vector<VarPtr>{}, if_stmt->span_));
            } else {
              result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, std::move(then_body),
                                                        std::nullopt, std::vector<VarPtr>{},
                                                        if_stmt->span_));
            }
          } else {
            StmtPtr else_body = MakeSeqOrSingle(std::move(lowered_remaining), if_stmt->span_);
            result.push_back(std::make_shared<IfStmt>(if_stmt->condition_, std::move(then_body),
                                                      std::make_optional(std::move(else_body)),
                                                      std::vector<VarPtr>{}, if_stmt->span_));
          }
        } else {
          // Pattern: if (cond) { continue }; remaining...  (bare continue, no pre_stmts)
          // → if (!cond) { remaining } [else { yield current_iter_args }]
          // No variable-update issue here since the then-body is empty.
          std::optional<StmtPtr> else_body;
          if (!iter_args.empty()) {
            std::vector<ExprPtr> yield_values;
            for (const auto& ia : iter_args) yield_values.push_back(ia);
            else_body = std::make_shared<YieldStmt>(std::move(yield_values), if_stmt->span_);
          }

          if (lowered_remaining.empty()) {
            if (!iter_args.empty()) {
              std::vector<ExprPtr> yield_values;
              for (const auto& ia : iter_args) yield_values.push_back(ia);
              StmtPtr guard_body = std::make_shared<YieldStmt>(std::move(yield_values), if_stmt->span_);
              ExprPtr negated_cond = NegateCondition(if_stmt->condition_, if_stmt->span_);
              result.push_back(std::make_shared<IfStmt>(std::move(negated_cond), std::move(guard_body),
                                                        std::move(else_body), std::vector<VarPtr>{},
                                                        if_stmt->span_));
            }
            // No remaining and no iter_args — drop the bare continue entirely.
          } else {
            StmtPtr guard_body = MakeSeqOrSingle(std::move(lowered_remaining), if_stmt->span_);
            ExprPtr negated_cond = NegateCondition(if_stmt->condition_, if_stmt->span_);
            result.push_back(std::make_shared<IfStmt>(std::move(negated_cond), std::move(guard_body),
                                                      std::move(else_body), if_stmt->return_vars_,
                                                      if_stmt->span_));
          }
        }

        return result;  // All remaining statements have been absorbed
      }

      // Check for `if (cond) { body } else { continue }`
      if (if_stmt->else_body_.has_value()) {
        auto else_stmts = FlattenToStmtList(*if_stmt->else_body_);
        bool else_is_continue = (else_stmts.size() == 1 && As<ContinueStmt>(else_stmts[0]) != nullptr);

        if (else_is_continue) {
          // Pattern: if (cond) { body } else { continue }; remaining...
          // → if (cond) { body; remaining... } [else { yield current_iter_args }]
          std::vector<StmtPtr> remaining(stmts.begin() + i + 1, stmts.end());
          auto lowered_remaining = LowerContinueInStmtList(remaining, iter_args);

          // Merge body + remaining into the then branch
          auto body_stmts = FlattenToStmtList(if_stmt->then_body_);
          // Recursively lower continue in the body too
          auto lowered_body = LowerContinueInStmtList(body_stmts, iter_args);
          for (auto& r : lowered_remaining) {
            lowered_body.push_back(std::move(r));
          }

          StmtPtr then_body = MakeSeqOrSingle(std::move(lowered_body), if_stmt->span_);

          std::optional<StmtPtr> else_body;
          if (!iter_args.empty()) {
            std::vector<ExprPtr> yield_values;
            for (const auto& ia : iter_args) {
              yield_values.push_back(ia);
            }
            else_body = std::make_shared<YieldStmt>(std::move(yield_values), if_stmt->span_);
          }

          auto new_if = std::make_shared<IfStmt>(if_stmt->condition_, std::move(then_body),
                                                 std::move(else_body), if_stmt->return_vars_, if_stmt->span_);
          result.push_back(new_if);
          return result;
        }
      }
    }

    // Check for bare continue statement (not inside an if)
    if (As<ContinueStmt>(stmt) != nullptr) {
      // Bare continue — everything after this is dead code
      if (!iter_args.empty()) {
        std::vector<ExprPtr> yield_values;
        for (const auto& ia : iter_args) {
          yield_values.push_back(ia);
        }
        result.push_back(std::make_shared<YieldStmt>(std::move(yield_values), stmt->span_));
      }
      return result;  // Drop remaining statements
    }

    // Not a continue pattern — recurse into the statement and add it
    result.push_back(LowerContinueInStmt(stmt, iter_args));
  }

  return result;
}

// ============================================================================
// Break Lowering (Value-Producing)
// ============================================================================

/// Lower break statements in a list of statements to value-producing IfStmts.
///
/// Each `if (cond) { break }` pattern is transformed into a value-producing
/// IfStmt that yields (bool_flag, updated_iter_args...) — where bool_flag is
/// true on the break path and false on the normal path.  This allows the
/// surrounding scf.for / scf.while to correctly propagate the break result
/// as an iter_arg without relying on invalid scf.break / scf.continue ops.
///
/// Returns:
///   new_stmts:    transformed statements (may end with a value-producing IfStmt)
///   result_exprs: 1 + N values: [bool_flag, ia_0_update, ..., ia_N_update]
///
/// No-break case:  result_exprs = [ConstBool(false), trailing_yield_values...]
/// Break found:    result_exprs = Var(s) produced by the innermost break IfStmt
static std::pair<std::vector<StmtPtr>, std::vector<ExprPtr>> LowerBreakToValue(
    const std::vector<StmtPtr>& stmts, const std::vector<IterArgPtr>& iter_args, int& var_counter,
    const Span& span) {
  for (size_t i = 0; i < stmts.size(); ++i) {
    const auto& stmt = stmts[i];

    // Determine if this statement is a break site and what the condition is.
    ExprPtr break_cond;

    // Case 1: bare BreakStmt — unconditional break
    if (As<BreakStmt>(stmt) != nullptr) {
      break_cond = std::make_shared<ConstBool>(true, span);
    }

    // Case 2: if (cond) { break }  (no else branch)
    if (!break_cond) {
      if (auto if_stmt = As<IfStmt>(stmt)) {
        auto then_stmts = FlattenToStmtList(if_stmt->then_body_);
        if (then_stmts.size() == 1 && As<BreakStmt>(then_stmts[0]) != nullptr &&
            !if_stmt->else_body_.has_value()) {
          break_cond = if_stmt->condition_;
        }
      }
    }

    if (!break_cond) continue;

    // Found a break site at index i.
    std::vector<StmtPtr> pre_stmts(stmts.begin(), stmts.begin() + i);
    std::vector<StmtPtr> rest(stmts.begin() + i + 1, stmts.end());

    // Recursively lower the remaining statements (handles multiple breaks).
    auto [rest_stmts, rest_exprs] = LowerBreakToValue(rest, iter_args, var_counter, span);

    // Create result vars for this break site: [_r_cont, _r_ia0, ..., _r_iaN]
    auto bool_type = std::make_shared<ScalarType>(DataType::BOOL);
    std::vector<VarPtr> result_vars;
    std::vector<ExprPtr> result_exprs;

    auto r_brk = std::make_shared<Var>("_cont_r" + std::to_string(var_counter++), bool_type, span);
    result_vars.push_back(r_brk);
    result_exprs.push_back(r_brk);
    for (size_t j = 0; j < iter_args.size(); ++j) {
      auto r_ia = std::make_shared<Var>(
          "_cont_ia" + std::to_string(var_counter++) + "_" + std::to_string(j),
          iter_args[j]->GetType(), span);
      result_vars.push_back(r_ia);
      result_exprs.push_back(r_ia);
    }

    // Then-branch (break path): yield(false, current_iter_args...)
    // false = cannot continue (break was taken)
    std::vector<ExprPtr> break_yield;
    break_yield.push_back(std::make_shared<ConstBool>(false, span));
    for (const auto& ia : iter_args) break_yield.push_back(ia);
    StmtPtr then_body = std::make_shared<YieldStmt>(std::move(break_yield), span);

    // Else-branch (normal path): rest_stmts + YieldStmt(rest_exprs)
    rest_stmts.push_back(std::make_shared<YieldStmt>(rest_exprs, span));
    StmtPtr else_body = MakeSeqOrSingle(std::move(rest_stmts), span);

    // Value-producing IfStmt for this break site
    auto inner_if = std::make_shared<IfStmt>(std::move(break_cond), std::move(then_body),
                                             std::optional<StmtPtr>(std::move(else_body)),
                                             std::move(result_vars), span);

    std::vector<StmtPtr> new_stmts = std::move(pre_stmts);
    new_stmts.push_back(inner_if);
    return {std::move(new_stmts), std::move(result_exprs)};
  }

  // No break found: return [true, trailing_yield_values...]
  // true = can continue (no break taken)
  std::vector<ExprPtr> no_break_exprs;
  no_break_exprs.push_back(std::make_shared<ConstBool>(true, span));

  std::vector<StmtPtr> no_break_stmts;
  if (!stmts.empty()) {
    if (auto trailing = As<YieldStmt>(stmts.back())) {
      no_break_stmts = std::vector<StmtPtr>(stmts.begin(), stmts.end() - 1);
      for (const auto& v : trailing->value_) no_break_exprs.push_back(v);
    } else {
      no_break_stmts = stmts;
    }
  }

  return {std::move(no_break_stmts), std::move(no_break_exprs)};
}

/// Lower break in a for loop by adding a _can_continue iter_arg.
///
/// Pattern:
///   for i in range(start, stop, step):
///     if (cond) { break }
///     body
///
/// Becomes (conceptually):
///   for i in range(start, stop, step) iter_args(_cont=true, ...):
///     _new_cont = if (_cont) {          // guard: skip body if already broken
///       _r = if (cond) {               // break site — value-producing
///         yield(false, iter_args...)   //   break path: cannot continue
///       } else {
///         body; yield(true, ...)       //   normal path: can continue
///       }
///       yield(_r, ...)
///     } else {
///       yield(_cont, ...)             // pass-through
///     }
///     yield(_new_cont, ...)           // updates for loop iter_arg
ForStmtPtr LowerBreakInFor(const ForStmtPtr& op) {
  Span span = op->span_;
  int var_counter = 0;

  // _can_continue iter_arg (init = true): true means the loop can continue, false means break was taken
  auto brk_type = std::make_shared<ScalarType>(DataType::BOOL);
  auto brk_iter_arg =
      std::make_shared<IterArg>("_can_continue", brk_type, std::make_shared<ConstBool>(true, span), span);
  auto brk_return_var = std::make_shared<Var>("_can_continue_final", brk_type, span);

  // new iter_args: [_can_continue, ...original]
  std::vector<IterArgPtr> new_iter_args = {brk_iter_arg};
  for (const auto& ia : op->iter_args_) new_iter_args.push_back(ia);

  // new return_vars: [_can_continue_final, ...original]
  std::vector<VarPtr> new_return_vars = {brk_return_var};
  for (const auto& rv : op->return_vars_) new_return_vars.push_back(rv);

  // Lower break sites to value-producing IfStmts.
  // result_exprs[0] = bool flag (false = break taken, true = can continue), result_exprs[1..N] = iter_arg updates.
  auto body_stmts = FlattenToStmtList(op->body_);
  auto [lowered, result_exprs] = LowerBreakToValue(body_stmts, op->iter_args_, var_counter, span);

  // Guard then-branch: lowered stmts + yield(result_exprs)
  lowered.push_back(std::make_shared<YieldStmt>(result_exprs, span));
  StmtPtr then_body = MakeSeqOrSingle(std::move(lowered), span);

  // Guard else-branch (pass-through when _can_continue is already false):
  // yield(_can_continue, ...iter_args...)
  std::vector<ExprPtr> skip_yield = {brk_iter_arg};
  for (const auto& ia : op->iter_args_) skip_yield.push_back(ia);
  StmtPtr else_body = std::make_shared<YieldStmt>(std::move(skip_yield), span);

  // Guard result vars: [_guard_brk, _guard_ia0, ...]
  std::vector<VarPtr> guard_vars;
  std::vector<ExprPtr> for_yield_exprs;

  auto guard_brk = std::make_shared<Var>("_guard_brk_" + std::to_string(var_counter++), brk_type, span);
  guard_vars.push_back(guard_brk);
  for_yield_exprs.push_back(guard_brk);
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    auto g = std::make_shared<Var>(
        "_guard_ia" + std::to_string(var_counter++) + "_" + std::to_string(i),
        op->iter_args_[i]->GetType(), span);
    guard_vars.push_back(g);
    for_yield_exprs.push_back(g);
  }

  // Guard: if (_can_continue) { then_body } else { else_body }
  // No NOT needed — _can_continue is already false when break was taken
  auto guard_if = std::make_shared<IfStmt>(brk_iter_arg, std::move(then_body),
                                           std::optional<StmtPtr>(std::move(else_body)),
                                           std::move(guard_vars), span);

  // For loop body: [guard_if, yield(for_yield_exprs)] — the trailing yield feeds scf.for's scf.yield
  auto for_body = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{guard_if, std::make_shared<YieldStmt>(for_yield_exprs, span)}, span);

  return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, std::move(new_iter_args),
                                   for_body, std::move(new_return_vars), span, op->kind_, op->chunk_size_,
                                   op->chunk_policy_, op->loop_origin_);
}

/// Lower break in a while loop using a do-region original-condition check.
///
/// Avoids And/Or/Not: the before-region checks only _can_continue; the original
/// condition is checked at the start of the do-region via a value-producing scf.if.
///
/// Pattern:
///   while (cond):
///     if (brk_cond): break
///     body
///
/// Becomes (conceptually):
///   while (_can_continue):                    // before: trivial flag check, no And
///     if (cond):                              // do: check original condition first
///       _r = if (brk_cond) {                 //   break site — value-producing
///         yield(false, iter_args...)          //     break path: cannot continue
///       } else {
///         body; yield(true, ...)             //     normal path: can continue
///       }
///       yield(_r, ...)
///     else:
///       yield(false, iter_args...)           // cond false → stop loop
WhileStmtPtr LowerBreakInWhile(const WhileStmtPtr& op) {
  Span span = op->span_;
  int var_counter = 0;

  auto brk_type = std::make_shared<ScalarType>(DataType::BOOL);
  // _can_continue iter_arg (init = true): true means the loop can continue, false means break was taken
  auto brk_iter_arg =
      std::make_shared<IterArg>("_can_continue", brk_type, std::make_shared<ConstBool>(true, span), span);
  auto brk_return_var = std::make_shared<Var>("_can_continue_final", brk_type, span);

  // new iter_args: [_can_continue, ...original]
  std::vector<IterArgPtr> new_iter_args = {brk_iter_arg};
  for (const auto& ia : op->iter_args_) new_iter_args.push_back(ia);

  std::vector<VarPtr> new_return_vars = {brk_return_var};
  for (const auto& rv : op->return_vars_) new_return_vars.push_back(rv);

  // Condition: just _can_continue — no And needed.
  // The original condition is checked inside the do-region.
  ExprPtr new_condition = brk_iter_arg;

  // Lower break sites to value-producing IfStmts.
  // result_exprs = [bool_flag (true=continue), ia_0_update, ..., ia_N_update]
  auto body_stmts = FlattenToStmtList(op->body_);
  auto [lowered, result_exprs] = LowerBreakToValue(body_stmts, op->iter_args_, var_counter, span);

  // Then-branch of outer if (original cond true): break-lowered body + yield(result_exprs)
  lowered.push_back(std::make_shared<YieldStmt>(result_exprs, span));
  StmtPtr then_body = MakeSeqOrSingle(std::move(lowered), span);

  // Else-branch of outer if (original cond false): yield(false, iter_args...) — loop terminates
  std::vector<ExprPtr> cond_false_yield;
  cond_false_yield.push_back(std::make_shared<ConstBool>(false, span));
  for (const auto& ia : op->iter_args_) cond_false_yield.push_back(ia);
  StmtPtr else_body = std::make_shared<YieldStmt>(std::move(cond_false_yield), span);

  // Outer result vars: [_outer_cont, _outer_ia0, ...] — one per (1 + N iter_args)
  std::vector<VarPtr> outer_vars;
  std::vector<ExprPtr> outer_exprs;
  auto outer_cont =
      std::make_shared<Var>("_outer_cont_" + std::to_string(var_counter++), brk_type, span);
  outer_vars.push_back(outer_cont);
  outer_exprs.push_back(outer_cont);
  for (size_t i = 0; i < op->iter_args_.size(); ++i) {
    auto ov = std::make_shared<Var>(
        "_outer_ia" + std::to_string(var_counter++) + "_" + std::to_string(i),
        op->iter_args_[i]->GetType(), span);
    outer_vars.push_back(ov);
    outer_exprs.push_back(ov);
  }

  // Outer if: if (orig_cond) { then_body } else { else_body } → produces outer_vars
  auto outer_if = std::make_shared<IfStmt>(op->condition_, std::move(then_body),
                                           std::optional<StmtPtr>(std::move(else_body)),
                                           std::move(outer_vars), span);

  // New body: [outer_if, trailing_yield(outer_exprs)] — trailing yield feeds scf.while scf.yield
  auto new_body = std::make_shared<SeqStmts>(
      std::vector<StmtPtr>{outer_if, std::make_shared<YieldStmt>(outer_exprs, span)}, span);

  return std::make_shared<WhileStmt>(std::move(new_condition), std::move(new_iter_args), new_body,
                                     std::move(new_return_vars), span);
}

// ============================================================================
// Main Pass Mutator
// ============================================================================

class LowerBreakContinueMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // First, recursively lower break/continue in nested structures
    StmtPtr new_body = VisitStmt(op->body_);

    bool has_break = ContainsBreak(new_body);
    bool has_continue = ContainsContinue(new_body);

    if (!has_break && !has_continue) {
      if (new_body == op->body_) return op;
      return std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, op->iter_args_,
                                       std::move(new_body), op->return_vars_, op->span_, op->kind_,
                                       op->chunk_size_, op->chunk_policy_, op->loop_origin_);
    }

    // Create a working copy with recursed body (use ForStmtPtr = shared_ptr<const ForStmt>)
    ForStmtPtr working = std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_,
                                                   op->iter_args_, std::move(new_body), op->return_vars_,
                                                   op->span_, op->kind_, op->chunk_size_, op->chunk_policy_,
                                                   op->loop_origin_);

    // Lower continue first (simpler transformation)
    if (has_continue) {
      auto body_stmts = FlattenToStmtList(working->body_);
      auto lowered = LowerContinueInStmtList(body_stmts, working->iter_args_);
      auto lowered_body = MakeSeqOrSingle(std::move(lowered), working->span_);
      working = ForStmtPtr(std::make_shared<ForStmt>(
          working->loop_var_, working->start_, working->stop_, working->step_, working->iter_args_,
          std::move(lowered_body), working->return_vars_, working->span_, working->kind_,
          working->chunk_size_, working->chunk_policy_, working->loop_origin_));
    }

    // Lower break (adds iter_arg, wraps body)
    if (has_break) {
      working = LowerBreakInFor(working);
    }

    return working;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    // First, recursively lower break/continue in nested structures
    StmtPtr new_body = VisitStmt(op->body_);

    bool has_break = ContainsBreak(new_body);
    bool has_continue = ContainsContinue(new_body);

    if (!has_break && !has_continue) {
      if (new_body == op->body_) return op;
      return std::make_shared<WhileStmt>(op->condition_, op->iter_args_, std::move(new_body), op->return_vars_,
                                         op->span_);
    }

    WhileStmtPtr working = std::make_shared<WhileStmt>(op->condition_, op->iter_args_, std::move(new_body),
                                                       op->return_vars_, op->span_);

    if (has_continue) {
      auto body_stmts = FlattenToStmtList(working->body_);
      auto lowered = LowerContinueInStmtList(body_stmts, working->iter_args_);
      auto lowered_body = MakeSeqOrSingle(std::move(lowered), working->span_);
      working = WhileStmtPtr(std::make_shared<WhileStmt>(working->condition_, working->iter_args_,
                                                         std::move(lowered_body), working->return_vars_,
                                                         working->span_));
    }

    if (has_break) {
      working = LowerBreakInWhile(working);
    }

    return working;
  }
};

}  // namespace

// ============================================================================
// Public API
// ============================================================================

FunctionPtr LowerBreakContinueImpl(const FunctionPtr& func) {
  if (!func || !func->body_) return func;

  // Run the mutator unconditionally — it internally checks for break/continue
  // at each loop level and short-circuits when none are present.
  // (ContainsBreak/ContainsContinue skip into nested loops, so they cannot
  //  be used here to pre-check the function body which may itself be a loop.)
  LowerBreakContinueMutator mutator;
  StmtPtr new_body = mutator.VisitStmt(func->body_);

  if (new_body == func->body_) return func;

  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                    func->return_types_, std::move(new_body), func->span_,
                                    func->func_type_);
}

namespace pass {

Pass LowerBreakContinue() {
  return CreateFunctionPass(LowerBreakContinueImpl, "LowerBreakContinue", kLowerBreakContinueProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
