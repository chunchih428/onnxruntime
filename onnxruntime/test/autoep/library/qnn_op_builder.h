// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>
#include "example_plugin_ep_utils.h"

namespace qnn_ep {

/// <summary>
/// Interface for QNN op builders (following QNN provider pattern)
/// </summary>
class IOpBuilder {
 public:
  virtual ~IOpBuilder() = default;

  /// <summary>
  /// Check whether the operator is supported
  /// </summary>
  virtual OrtStatus* IsOpSupported(const ApiPtrs& apis, const OrtNode* node) const = 0;

  /// <summary>
  /// Get the op builder type for identification
  /// </summary>
  virtual std::string GetOpBuilderType() const = 0;
};

/// <summary>
/// Base implementation providing common functionality for all op builders
/// </summary>
class BaseOpBuilder : public IOpBuilder {
 public:
  BaseOpBuilder(const std::string& op_builder_type) : op_builder_type_(op_builder_type) {}

  std::string GetOpBuilderType() const override { return op_builder_type_; }

 protected:
  /// <summary>
  /// Validate input/output counts for the node
  /// </summary>
  OrtStatus* ValidateInputOutputCounts(const ApiPtrs& apis, const OrtNode* node,
                                       size_t expected_inputs, size_t expected_outputs) const;

  /// <summary>
  /// Check if all inputs and outputs are float tensors
  /// </summary>
  OrtStatus* ValidateFloatTensors(const ApiPtrs& apis, const OrtNode* node) const;

 private:
  const std::string op_builder_type_;
};

/// <summary>
/// Simple op builder for basic operators like Mul, Add, etc.
/// </summary>
class SimpleOpBuilder : public BaseOpBuilder {
 public:
  SimpleOpBuilder() : BaseOpBuilder("SimpleOpBuilder") {}

  OrtStatus* IsOpSupported(const ApiPtrs& apis, const OrtNode* node) const override;

 private:
  /// <summary>
  /// Get expected input/output counts for simple ops
  /// </summary>
  bool GetExpectedCounts(const std::string& op_type, size_t& expected_inputs, size_t& expected_outputs) const;
};

}  // namespace qnn_ep