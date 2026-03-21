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

#include "pypto/backend/common/cann_runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "pypto/core/logging.h"

namespace pypto {
namespace backend {

static constexpr uint32_t kBufSize = 64;

CannRuntime::CannRuntime() {
  // Try ASCEND_CANN_PACKAGE_PATH first, then fall back to system library path
  const char* cann_path = std::getenv("ASCEND_CANN_PACKAGE_PATH");
  if (cann_path != nullptr) {
    std::string so_path = std::string(cann_path) + "/lib64/libruntime.so";
    handle_ = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  if (handle_ == nullptr) {
    handle_ = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
  }
  if (handle_ != nullptr) {
    soc_spec_func_ = reinterpret_cast<GetSocSpecFunc>(dlsym(handle_, "rtGetSocSpec"));
  }
  if (handle_ == nullptr) {
    LOG_WARN << "CANN libruntime.so not found; SoC specs will use hardcoded defaults.";
  } else if (soc_spec_func_ == nullptr) {
    LOG_WARN << "rtGetSocSpec not found in libruntime.so; SoC specs will use hardcoded defaults.";
  }
}

CannRuntime::~CannRuntime() {
  if (handle_ != nullptr) {
    dlclose(handle_);
  }
}

CannRuntime& CannRuntime::Instance() {
  static CannRuntime instance;
  return instance;
}

bool CannRuntime::GetSocSpec(const std::string& column, const std::string& key, std::string& val) const {
  if (soc_spec_func_ == nullptr) {
    return false;
  }
  char buf[kBufSize] = {};
  int ret = soc_spec_func_(column.c_str(), key.c_str(), buf, kBufSize);
  if (ret != 0) {
    return false;
  }
  buf[kBufSize - 1] = '\0';
  val = std::string(buf);
  return true;
}

}  // namespace backend
}  // namespace pypto
