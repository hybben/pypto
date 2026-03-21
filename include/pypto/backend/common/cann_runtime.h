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

#ifndef PYPTO_BACKEND_COMMON_CANN_RUNTIME_H_
#define PYPTO_BACKEND_COMMON_CANN_RUNTIME_H_

#include <cstdint>
#include <string>

namespace pypto {
namespace backend {

/**
 * @brief Thin wrapper around CANN's rtGetSocSpec, loaded at runtime via dlopen.
 *
 * No CANN headers are required at build time. When CANN is unavailable,
 * IsAvailable() returns false and GetSocSpec() always returns false.
 */
class CannRuntime {
 public:
  static CannRuntime& Instance();

  CannRuntime(const CannRuntime&) = delete;
  CannRuntime& operator=(const CannRuntime&) = delete;

  /** Returns true if libruntime.so was loaded and rtGetSocSpec was resolved. */
  bool IsAvailable() const { return soc_spec_func_ != nullptr; }

  /**
   * @brief Query a SoC spec value.
   *
   * @param column  Section name, e.g. "SoCInfo" or "AICoreSpec"
   * @param key     Key name, e.g. "cube_core_cnt" or "l1_size"
   * @param val     Output string value on success
   * @return true if the query succeeded (rtGetSocSpec returned 0)
   */
  bool GetSocSpec(const std::string& column, const std::string& key, std::string& val) const;

 private:
  CannRuntime();
  ~CannRuntime();

  using GetSocSpecFunc = int (*)(const char*, const char*, char*, uint32_t);
  GetSocSpecFunc soc_spec_func_ = nullptr;
  void* handle_ = nullptr;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_COMMON_CANN_RUNTIME_H_
