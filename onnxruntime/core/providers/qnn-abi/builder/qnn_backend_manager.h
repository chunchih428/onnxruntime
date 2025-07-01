// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>

#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

class QnnBackendManager {
 public:
  QnnBackendManager() = default;
  ~QnnBackendManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnBackendManager);

  // Basic methods for SharedContext compatibility
  const std::string& GetQnnBackendType() const { return backend_type_; }

 private:
  std::string backend_type_ = "htp";
};

}  // namespace qnn
}  // namespace onnxruntime