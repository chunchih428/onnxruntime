// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <vector>
#include <string>
#include <memory>

#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations
class QnnBackendManager;

struct QnnTensorInfo {
  uint32_t tensor_byte_size = 0;
  size_t ort_index = 0;
};

class QnnModel {
 public:
  QnnModel(QnnBackendManager* qnn_backend_manager, const std::string& name)
      : qnn_backend_manager_(qnn_backend_manager), name_(name) {
  }

  ~QnnModel() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(QnnModel);

  const std::string& Name() const { return name_; }

  // Status methods for compatibility with SharedContext usage patterns
  Status SetGraphInputOutputInfo(const GraphViewer& graph_viewer,
                                 const Node& fused_node,
                                 const logging::Logger& logger) {
    // Implementation would set up input/output info
    return Status::OK();
  }

  Status SetupQnnInputOutput(const logging::Logger& logger) {
    // Implementation would setup QNN input/output tensors
    return Status::OK();
  }

  Status ExecuteGraph(const logging::Logger& logger) {
    // Implementation would execute the QNN graph
    return Status::OK();
  }

 private:
  QnnBackendManager* qnn_backend_manager_;
  std::string name_;
};

}  // namespace qnn
}  // namespace onnxruntime