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

/**
 * @file ptr_ops.cpp
 * @brief Pointer operations for ptoas IR scene (ptr.addptr, ptr.make_tensor)
 *
 * These ops emit PTO MLIR instructions (pto.addptr, pto.make_tensor_view) and
 * are distinct from the orchestration-only tensor ops.
 */

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

TypePtr DeduceAddPtrType(const std::vector<ExprPtr>& args,
                         const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // ptr.addptr: Advance a pointer by an integer offset
  // Args: (ptr, offset)
  // Returns: same PtrType as input (pointer bumped but same element dtype)
  CHECK(args.size() == 2) << "ptr.addptr requires exactly 2 arguments (ptr, offset), but got "
                          << args.size();

  // First argument must be PtrType
  auto ptr_type = As<PtrType>(args[0]->GetType());
  CHECK(ptr_type) << "ptr.addptr requires first argument to be a PtrType, but got "
                  << args[0]->GetType()->TypeName()
                  << ". Use pl.Ptr[dtype] to annotate pointer parameters.";

  // Second argument must be ScalarType with integer or index dtype
  auto offset_type = As<ScalarType>(args[1]->GetType());
  CHECK(offset_type) << "ptr.addptr requires second argument (offset) to be a ScalarType, but got "
                     << args[1]->GetType()->TypeName();
  CHECK(offset_type->dtype_.IsInt() || offset_type->dtype_ == DataType(DataType::INDEX))
      << "ptr.addptr offset must have integer or index dtype, but got "
      << offset_type->dtype_.ToString();

  // Return the same PtrType (pointer is advanced but still points to same element type),
  // with base_ptr/offset annotations for codegen indirect-select support.
  ExprPtr new_base_ptr;
  ExprPtr new_offset;

  if (ptr_type->base_ptr.has_value()) {
    // Chained addptr: propagate base from the input ptr, fold offsets if possible.
    new_base_ptr = *ptr_type->base_ptr;
    if (auto c1 = As<ConstInt>(*ptr_type->offset)) {
      if (auto c2 = As<ConstInt>(args[1])) {
        new_offset = std::make_shared<ConstInt>(c1->value_ + c2->value_,
                                                DataType(DataType::INDEX), args[1]->span_);
      } else {
        new_offset = std::make_shared<Add>(*ptr_type->offset, args[1],
                                           DataType(DataType::INDEX), args[1]->span_);
      }
    } else {
      new_offset = std::make_shared<Add>(*ptr_type->offset, args[1],
                                         DataType(DataType::INDEX), args[1]->span_);
    }
  } else {
    // Direct addptr on a function parameter — record base and offset directly.
    new_base_ptr = args[0];
    new_offset   = args[1];
  }

  return std::make_shared<PtrType>(ptr_type->dtype_, new_base_ptr, new_offset);
}

REGISTER_OP("ptr.addptr")
    .set_op_category("PtrOp")
    .set_description("Advance a pointer by an integer offset (emits pto.addptr)")
    .add_argument("ptr", "Input raw pointer (PtrType)")
    .add_argument("offset", "Integer byte offset (ScalarType with integer/index dtype)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceAddPtrType(args, kwargs);
    });

TypePtr DeduceMakeTensorType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs) {
  // ptr.make_tensor: Create a tensor view from a pointer with explicit shape and strides
  // Args: (ptr, shape_tuple, stride_tuple)
  CHECK(args.size() == 3) << "ptr.make_tensor requires exactly 3 arguments (ptr, shape, stride), but got "
                          << args.size();

  // First argument must be PtrType (a raw pointer to typed global memory)
  auto ptr_type = As<PtrType>(args[0]->GetType());
  CHECK(ptr_type) << "ptr.make_tensor requires first argument to be a PtrType, but got "
                  << args[0]->GetType()->TypeName()
                  << ". Use pl.Ptr[dtype] to annotate pointer parameters.";

  // Second argument must be MakeTuple (shape)
  auto shape_tuple = As<MakeTuple>(args[1]);
  CHECK(shape_tuple) << "ptr.make_tensor requires shape to be a MakeTuple";

  // Third argument must be MakeTuple (stride)
  auto stride_tuple = As<MakeTuple>(args[2]);
  CHECK(stride_tuple) << "ptr.make_tensor requires stride to be a MakeTuple";

  CHECK(shape_tuple->elements_.size() == stride_tuple->elements_.size())
      << "ptr.make_tensor shape rank (" << shape_tuple->elements_.size()
      << ") must match stride rank (" << stride_tuple->elements_.size() << ")";

  TensorView tv(stride_tuple->elements_, TensorLayout::ND, args[0]);
  return std::make_shared<TensorType>(shape_tuple->elements_, ptr_type->dtype_, std::nullopt, tv);
}

REGISTER_OP("ptr.make_tensor")
    .set_op_category("PtrOp")
    .set_description("Create a tensor view from a pointer with explicit shape and strides"
                     " (emits pto.make_tensor_view)")
    .add_argument("ptr", "Input raw pointer (PtrType)")
    .add_argument("shape", "New shape dimensions (MakeTuple of ConstInt)")
    .add_argument("stride", "Stride per dimension (MakeTuple of ConstInt or Var)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceMakeTensorType(args, kwargs);
    });

}  // namespace ir
}  // namespace pypto
