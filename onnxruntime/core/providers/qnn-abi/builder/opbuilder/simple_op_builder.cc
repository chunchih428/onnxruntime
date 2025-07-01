// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/opbuilder/simple_op_builder.h"
#include <unordered_map>

namespace qnn_ep {

OrtStatus* SimpleOpBuilder::IsOpSupported(const ApiPtrs& apis, const OrtNode* node) const {
  const char* op_type = nullptr;
  apis.ort_api.Node_GetOperatorType(node, &op_type);
  
  size_t expected_inputs, expected_outputs;
  if (!GetExpectedCounts(op_type, expected_inputs, expected_outputs)) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported operation type");
  }
  
  // Validate input/output counts
  OrtStatus* status = ValidateInputOutputCounts(apis, node, expected_inputs, expected_outputs);
  if (status != nullptr) {
    return status;
  }
  
  // Validate that all inputs and outputs are float tensors for most ops
  return ValidateFloatTensors(apis, node);
}

OrtStatus* SimpleOpBuilder::ProcessAttributesAndOutputs(const ApiPtrs& apis, const OrtNode* node,
                                                        void* qnn_model_wrapper) const {
  // Simple ops typically don't need special attribute processing
  return ProcessOutputs(apis, node, qnn_model_wrapper);
}

bool SimpleOpBuilder::GetExpectedCounts(const std::string& op_type, size_t& expected_inputs, size_t& expected_outputs) const {
  static const std::unordered_map<std::string, std::pair<size_t, size_t>> op_counts = {
      // Binary ops
      {"Add", {2, 1}},
      {"Mul", {2, 1}},
      {"Sub", {2, 1}},
      {"Div", {2, 1}},
      {"Max", {2, 1}},
      {"Min", {2, 1}},
      {"Pow", {2, 1}},
      {"Equal", {2, 1}},
      {"Greater", {2, 1}},
      {"GreaterOrEqual", {2, 1}},
      {"Less", {2, 1}},
      {"LessOrEqual", {2, 1}},
      {"And", {2, 1}},
      {"Or", {2, 1}},
      {"Where", {3, 1}},
      
      // Unary ops
      {"Abs", {1, 1}},
      {"Asin", {1, 1}},
      {"Atan", {1, 1}},
      {"Ceil", {1, 1}},
      {"Sign", {1, 1}},
      {"Cast", {1, 1}},
      {"Cos", {1, 1}},
      {"Exp", {1, 1}},
      {"Floor", {1, 1}},
      {"Log", {1, 1}},
      {"Neg", {1, 1}},
      {"Not", {1, 1}},
      {"Round", {1, 1}},
      {"Sigmoid", {1, 1}},
      {"Sin", {1, 1}},
      {"Sqrt", {1, 1}},
      {"Tanh", {1, 1}},
      {"Relu", {1, 1}},
      {"Gelu", {1, 1}},
      {"Elu", {1, 1}},
      {"HardSigmoid", {1, 1}},
      {"HardSwish", {1, 1}},
      {"Softmax", {1, 1}},
      {"LogSoftmax", {1, 1}},
      {"Transpose", {1, 1}},
      {"Reshape", {2, 1}},
      {"Squeeze", {1, 1}},
      {"Unsqueeze", {1, 1}},
      {"Flatten", {1, 1}},
      {"DepthToSpace", {1, 1}},
      {"SpaceToDepth", {1, 1}},
      {"DequantizeLinear", {1, 1}},
      {"QuantizeLinear", {1, 1}},
      
      // Special ops with variable inputs
      {"Clip", {1, 1}},  // Can have 1-3 inputs
      {"PRelu", {2, 1}},
      {"LeakyRelu", {1, 1}},
      {"MatMul", {2, 1}},
      {"Gemm", {2, 1}},  // Can have 2-3 inputs
      {"Conv", {2, 1}},  // Can have 2-3 inputs
      {"ConvTranspose", {2, 1}},  // Can have 2-3 inputs
      
      // Pooling ops
      {"GlobalAveragePool", {1, 1}},
      {"AveragePool", {1, 1}},
      {"MaxPool", {1, 1}},
      {"GlobalMaxPool", {1, 1}},
      
      // Reduction ops
      {"ReduceMax", {1, 1}},
      {"ReduceMean", {1, 1}},
      {"ReduceMin", {1, 1}},
      {"ReduceProd", {1, 1}},
      {"ReduceSum", {1, 1}},
      
      // Other ops
      {"Gather", {2, 1}},
      {"GatherElements", {2, 1}},
      {"ScatterND", {3, 1}},
      {"Slice", {1, 1}},  // Can have 3-5 inputs
      {"Split", {1, 2}},  // Variable outputs
      {"Resize", {1, 1}},  // Can have 2-4 inputs
      {"Upsample", {1, 1}},  // Can have 2 inputs
      {"Concat", {2, 1}},  // Variable inputs
      {"CumSum", {2, 1}},
      {"ArgMax", {1, 1}},
      {"ArgMin", {1, 1}},
      {"Tile", {2, 1}},
      {"TopK", {2, 2}},
      {"InstanceNormalization", {1, 1}},  // Can have 2-3 inputs
      {"BatchNormalization", {1, 1}},  // Can have 3-5 inputs
      {"LayerNormalization", {1, 1}},  // Can have 2-3 inputs
      {"LRN", {1, 1}},
      {"Pad", {1, 1}},  // Can have 2-3 inputs
      {"Expand", {2, 1}},
      {"GridSample", {2, 1}},
      {"LpNormalization", {1, 1}},
      {"LSTM", {3, 3}},  // Variable inputs/outputs
      {"Sum", {2, 1}}  // Variable inputs
  };
  
  auto it = op_counts.find(op_type);
  if (it != op_counts.end()) {
    expected_inputs = it->second.first;
    expected_outputs = it->second.second;
    return true;
  }
  
  return false;
}

}  // namespace qnn_ep