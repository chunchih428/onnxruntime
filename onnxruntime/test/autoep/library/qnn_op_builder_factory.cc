// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_op_builder_factory.h"

namespace qnn_ep {

OpBuilderRegistrations::OpBuilderRegistrations() {
  // Register supported ops (following QNN provider pattern)
  
  // Element-wise operations
  AddOpBuilder("Mul", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Add", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Sub", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Div", std::make_unique<SimpleOpBuilder>());
  
  // Activation operations
  AddOpBuilder("Relu", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Sigmoid", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Tanh", std::make_unique<SimpleOpBuilder>());
  
  // Linear operations
  AddOpBuilder("Conv", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("MatMul", std::make_unique<SimpleOpBuilder>());
  AddOpBuilder("Gemm", std::make_unique<SimpleOpBuilder>());
}

void OpBuilderRegistrations::AddOpBuilder(const std::string& onnx_op_type, std::unique_ptr<IOpBuilder> builder) {
  supported_op_types_.insert(onnx_op_type);
  op_builder_map_[onnx_op_type] = builder.get();
  builders_.push_back(std::move(builder));
}

const IOpBuilder* OpBuilderRegistrations::GetOpBuilderByOnnxOpType(const std::string& onnx_op_type) const {
  auto it = op_builder_map_.find(onnx_op_type);
  return (it != op_builder_map_.end()) ? it->second : nullptr;
}

bool OpBuilderRegistrations::IsOpTypeSupported(const std::string& onnx_op_type) const {
  return supported_op_types_.find(onnx_op_type) != supported_op_types_.end();
}

// Global instance (following QNN provider pattern)
const OpBuilderRegistrations& GetOpBuilderRegistrations() {
  static const OpBuilderRegistrations registrations;
  return registrations;
}

const IOpBuilder* GetOpBuilder(const std::string& onnx_op_type) {
  return GetOpBuilderRegistrations().GetOpBuilderByOnnxOpType(onnx_op_type);
}

bool IsOpSupported(const std::string& onnx_op_type) {
  return GetOpBuilderRegistrations().IsOpTypeSupported(onnx_op_type);
}

}  // namespace qnn_ep