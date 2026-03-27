/*
 * Copyright (c) PyPTO Contributors.
 * CANN Open Software License Agreement Version 2.0.
 * -----------------------------------------------------------------------------------------------------------
 *
 * ConstFoldAndSimplify pass:
 *   1. Constant folding: ConstInt op ConstInt → ConstInt
 *   2. If-stmt simplification:
 *      - constant condition → inline taken branch
 *      - both branches yield same value → replace with that value
 *   3. Redundant assign: x = x → remove
 */

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// ---------------------------------------------------------------------------
// Helper: try to extract compile-time integer value from an Expr
// ---------------------------------------------------------------------------
static std::optional<int64_t> TryGetConstInt(const ExprPtr& expr) {
  if (!expr) return std::nullopt;
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  return std::nullopt;
}

static DataType GetDType(const ExprPtr& expr) {
  if (auto st = As<ScalarType>(expr->GetType())) return st->dtype_;
  return DataType::INDEX;
}

// ---------------------------------------------------------------------------
// Helper: resolve Var through a value map to find the underlying ConstInt
// ---------------------------------------------------------------------------
static std::optional<int64_t> ResolveToConst(const ExprPtr& expr,
                                              const std::unordered_map<std::string, ExprPtr>& var_vals) {
  if (!expr) return std::nullopt;
  if (auto ci = As<ConstInt>(expr)) return ci->value_;
  if (auto v = As<Var>(expr)) {
    auto it = var_vals.find(v->name_);
    if (it != var_vals.end()) return ResolveToConst(it->second, var_vals);
  }
  // Resolve TupleGetItemExpr → MakeTuple element
  if (auto tgi = As<TupleGetItemExpr>(expr)) {
    // Try to resolve the tuple itself
    ExprPtr tuple_expr = tgi->tuple_;
    if (auto tv = As<Var>(tuple_expr)) {
      auto it = var_vals.find(tv->name_);
      if (it != var_vals.end()) tuple_expr = it->second;
    }
    if (auto mt = As<MakeTuple>(tuple_expr)) {
      int idx = tgi->index_;
      if (idx >= 0 && idx < static_cast<int>(mt->elements_.size())) {
        return ResolveToConst(mt->elements_[idx], var_vals);
      }
    }
  }
  return std::nullopt;
}

static bool IsSameExpr(const ExprPtr& a, const ExprPtr& b,
                        const std::unordered_map<std::string, ExprPtr>& var_vals = {}) {
  if (!a || !b) return false;
  if (a.get() == b.get()) return true;
  // Resolve through value map
  auto av = ResolveToConst(a, var_vals);
  auto bv = ResolveToConst(b, var_vals);
  if (av && bv) return *av == *bv;
  // Both Var with same name
  auto va = As<Var>(a);
  auto vb = As<Var>(b);
  if (va && vb) return va->name_ == vb->name_;
  return false;
}

// ---------------------------------------------------------------------------
// ConstFoldMutator — IRMutator subclass that folds constants and simplifies ifs
// ---------------------------------------------------------------------------
class ConstFoldMutator : public IRMutator {
 public:
  using IRMutator::VisitExpr_;
  using IRMutator::VisitStmt_;

  // Track Var → assigned Expr for value resolution
  std::unordered_map<std::string, ExprPtr> var_vals_;

  // ---- Track assignments for value resolution ----
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_val = VisitExpr(op->value_);
    // Record the assigned value (works for ConstInt, MakeTuple, TupleGetItemExpr, etc.)
    var_vals_[op->var_->name_] = new_val;
    if (new_val.get() != op->value_.get())
      return std::make_shared<AssignStmt>(op->var_, new_val, op->span_);
    return op;
  }

  // ---- Constant fold binary expressions ----
  ExprPtr VisitExpr_(const AddPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv + *rv, GetDType(op), op->span_);
    // x + 0 → x
    if (rv && *rv == 0) return l;
    if (lv && *lv == 0) return r;
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Add>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const SubPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv - *rv, GetDType(op), op->span_);
    // x - 0 → x
    if (rv && *rv == 0) return l;
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Sub>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const MulPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv * *rv, GetDType(op), op->span_);
    // x * 1 → x
    if (rv && *rv == 1) return l;
    if (lv && *lv == 1) return r;
    // x * 0 → 0
    if ((rv && *rv == 0) || (lv && *lv == 0))
      return std::make_shared<ConstInt>(0, GetDType(op), op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Mul>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const FloorDivPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv && *rv != 0)
      return std::make_shared<ConstInt>(*lv / *rv, GetDType(op), op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<FloorDiv>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const FloorModPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv && *rv != 0)
      return std::make_shared<ConstInt>(*lv % *rv, GetDType(op), op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<FloorMod>(l, r, GetDType(op), op->span_);
    return op;
  }

  // ---- Constant fold comparisons ----
  ExprPtr VisitExpr_(const EqPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv == *rv ? 1 : 0, DataType::BOOL, op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Eq>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const LtPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv < *rv ? 1 : 0, DataType::BOOL, op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Lt>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const LePtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv <= *rv ? 1 : 0, DataType::BOOL, op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Le>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const GtPtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv > *rv ? 1 : 0, DataType::BOOL, op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Gt>(l, r, GetDType(op), op->span_);
    return op;
  }

  ExprPtr VisitExpr_(const GePtr& op) override {
    auto l = VisitExpr(op->left_);
    auto r = VisitExpr(op->right_);
    auto lv = TryGetConstInt(l), rv = TryGetConstInt(r);
    if (lv && rv)
      return std::make_shared<ConstInt>(*lv >= *rv ? 1 : 0, DataType::BOOL, op->span_);
    if (l.get() != op->left_.get() || r.get() != op->right_.get())
      return std::make_shared<Ge>(l, r, GetDType(op), op->span_);
    return op;
  }

  // ---- Constant fold unary Neg ----
  ExprPtr VisitExpr_(const NegPtr& op) override {
    auto operand = VisitExpr(op->operand_);
    auto ov = TryGetConstInt(operand);
    if (ov) return std::make_shared<ConstInt>(-*ov, GetDType(op), op->span_);
    if (operand.get() != op->operand_.get())
      return std::make_shared<Neg>(operand, GetDType(op), op->span_);
    return op;
  }

  // ---- If-stmt simplification ----
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto cond = VisitExpr(op->condition_);
    auto then_body = VisitStmt(op->then_body_);
    std::optional<StmtPtr> else_body;
    if (op->else_body_) else_body = VisitStmt(*op->else_body_);

    // Constant condition → inline taken branch
    auto cv = TryGetConstInt(cond);
    if (cv) {
      if (*cv != 0) return then_body;          // condition true → then
      if (else_body) return *else_body;         // condition false → else
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);  // false, no else → empty
    }

    // Both branches yield the same value → replace with unconditional assign
    if (else_body && !op->return_vars_.empty()) {
      // Extract yield values from both branches
      auto extract_yields = [](const StmtPtr& body) -> std::vector<ExprPtr> {
        // Direct YieldStmt
        if (auto ys = As<YieldStmt>(body)) return ys->value_;
        // SeqStmts wrapping a single YieldStmt
        if (auto seq = As<SeqStmts>(body)) {
          if (seq->stmts_.size() == 1) {
            if (auto ys = As<YieldStmt>(seq->stmts_[0])) return ys->value_;
          }
          // YieldStmt as last element
          if (!seq->stmts_.empty()) {
            if (auto ys = As<YieldStmt>(seq->stmts_.back())) return ys->value_;
          }
        }
        return {};
      };
      auto then_vals = extract_yields(then_body);
      auto else_vals = extract_yields(*else_body);
      // (Debug removed — see git history for diagnostic version)
      if (then_vals.size() == op->return_vars_.size() && else_vals.size() == op->return_vars_.size()) {
        bool all_same = true;
        for (size_t i = 0; i < then_vals.size(); ++i) {
          if (!IsSameExpr(then_vals[i], else_vals[i], var_vals_)) { all_same = false; break; }
        }
        if (all_same) {
          // Replace with: assign each return_var = yield_value (no branch needed)
          std::vector<StmtPtr> assigns;
          for (size_t i = 0; i < op->return_vars_.size(); ++i) {
            assigns.push_back(std::make_shared<AssignStmt>(op->return_vars_[i], then_vals[i], op->span_));
          }
          return std::make_shared<SeqStmts>(std::move(assigns), op->span_);
        }
      }
    }

    // Copy-on-write
    if (cond.get() != op->condition_.get() || then_body.get() != op->then_body_.get() ||
        (else_body && else_body->get() != op->else_body_->get())) {
      return std::make_shared<IfStmt>(cond, then_body, else_body, op->return_vars_, op->span_);
    }
    return op;
  }
};

// ---------------------------------------------------------------------------
// Pass factory
// ---------------------------------------------------------------------------
static FunctionPtr TransformConstFoldAndSimplify(const FunctionPtr& func) {
  if (!func || !func->body_) return func;
  ConstFoldMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                    func->return_types_, new_body, func->span_, func->func_type_);
}

namespace pass {

Pass ConstFoldAndSimplify() {
  return CreateFunctionPass(TransformConstFoldAndSimplify, "ConstFoldAndSimplify");
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
