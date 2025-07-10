// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>

#include "core/providers/qnn-abi/ort_api.h"
#pragma once

namespace onnxruntime {

// Forward declaration for QNN model - since we don't have the full QnnModel class for QNN-ABI
class QnnModelMock {
public:
    QnnModelMock(const std::string& name) : name_(name) {}
    const std::string& Name() const { return name_; }
private:
    std::string name_;
};

class SharedContext {
 public:
  static SharedContext& GetInstance() {
    static SharedContext instance_;
    return instance_;
  }

  bool HasSharedQnnModels() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return !shared_qnn_models_.empty();
  }

  bool HasQnnModel(const std::string& model_name) {
    auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                      [&model_name](const std::unique_ptr<QnnModelMock>& qnn_model) { return qnn_model->Name() == model_name; });
    return it != shared_qnn_models_.end();
  }

  // Mock implementation for QNN-ABI - simplified version
  void AddQnnModel(const std::string& model_name) {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_qnn_models_.push_back(std::make_unique<QnnModelMock>(model_name));
  }

  void ClearQnnModels() {
    const std::lock_guard<std::mutex> lock(mtx_);
    shared_qnn_models_.clear();
  }

 private:
  SharedContext() = default;
  ~SharedContext() = default;

  // Used for passing through QNN models (deserialized from context binary) across sessions
  std::vector<std::unique_ptr<QnnModelMock>> shared_qnn_models_;
  // Producer sessions can be in parallel
  // Consumer sessions have to be after producer sessions initialized
  std::mutex mtx_;
};

}  // namespace onnxruntime
