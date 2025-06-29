// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "qnn_op_builder.h"

namespace qnn_ep {

/// <summary>
/// Op builder registry and factory (following QNN provider pattern)
/// </summary>
class OpBuilderRegistrations {
 public:
  OpBuilderRegistrations();

  /// <summary>
  /// Get op builder for given ONNX op type
  /// </summary>
  const IOpBuilder* GetOpBuilderByOnnxOpType(const std::string& onnx_op_type) const;

  /// <summary>
  /// Check if op type is supported
  /// </summary>
  bool IsOpTypeSupported(const std::string& onnx_op_type) const;

  /// <summary>
  /// Get all supported op types
  /// </summary>
  const std::unordered_set<std::string>& GetSupportedOpTypes() const { return supported_op_types_; }

 private:
  void AddOpBuilder(const std::string& onnx_op_type, std::unique_ptr<IOpBuilder> builder);
  
  std::vector<std::unique_ptr<IOpBuilder>> builders_;
  std::unordered_map<std::string, const IOpBuilder*> op_builder_map_;
  std::unordered_set<std::string> supported_op_types_;
};

/// <summary>
/// Get the global op builder registry
/// </summary>
const OpBuilderRegistrations& GetOpBuilderRegistrations();

/// <summary>
/// Get op builder for given ONNX op type (main entry point)
/// </summary>
const IOpBuilder* GetOpBuilder(const std::string& onnx_op_type);

/// <summary>
/// Check if operation is supported by QNN EP
/// </summary>
bool IsOpSupported(const std::string& onnx_op_type);

}  // namespace qnn_ep