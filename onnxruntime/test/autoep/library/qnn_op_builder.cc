// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_op_builder.h"
#include <unordered_map>
#include <unordered_set>

namespace qnn_ep {

// Helper function to check if tensor is float (from existing code)
// extern OrtStatus* IsFloatTensor(const OrtApi& ort_api, const OrtValueInfo* value_info, bool& is_float);

OrtStatus* BaseOpBuilder::ValidateInputOutputCounts(const ApiPtrs& apis, const OrtNode* node,
                                                     size_t expected_inputs, size_t expected_outputs) const {
  OrtArrayOfConstObjects* inputs_array = nullptr;
  OrtArrayOfConstObjects* outputs_array = nullptr;

  auto cleanup = [&]() {
    if (inputs_array) apis.ort_api.ReleaseArrayOfConstObjects(inputs_array);
    if (outputs_array) apis.ort_api.ReleaseArrayOfConstObjects(outputs_array);
  };

  RETURN_IF_ERROR(apis.ort_api.Node_GetInputs(node, &inputs_array));
  RETURN_IF_ERROR(apis.ort_api.Node_GetOutputs(node, &outputs_array));

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetSize(inputs_array, &num_inputs));
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetSize(outputs_array, &num_outputs));

  cleanup();

  if (num_inputs != expected_inputs) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected number of inputs");
  }
  if (num_outputs != expected_outputs) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unexpected number of outputs");
  }

  return nullptr;
}

OrtStatus* BaseOpBuilder::ValidateFloatTensors(const ApiPtrs& apis, const OrtNode* node) const {
  OrtArrayOfConstObjects* inputs_array = nullptr;
  OrtArrayOfConstObjects* outputs_array = nullptr;

  auto cleanup = [&]() {
    if (inputs_array) apis.ort_api.ReleaseArrayOfConstObjects(inputs_array);
    if (outputs_array) apis.ort_api.ReleaseArrayOfConstObjects(outputs_array);
  };

  RETURN_IF_ERROR(apis.ort_api.Node_GetInputs(node, &inputs_array));
  RETURN_IF_ERROR(apis.ort_api.Node_GetOutputs(node, &outputs_array));

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetSize(inputs_array, &num_inputs));
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetSize(outputs_array, &num_outputs));

  const void* const* inputs_data = nullptr;
  const void* const* outputs_data = nullptr;
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetData(inputs_array, &inputs_data));
  RETURN_IF_ERROR(apis.ort_api.ArrayOfConstObjects_GetData(outputs_array, &outputs_data));

  // Check input data types
  for (size_t i = 0; i < num_inputs; ++i) {
    bool is_float = false;
    RETURN_IF_ERROR(IsFloatTensor(apis.ort_api, static_cast<const OrtValueInfo*>(inputs_data[i]), is_float));
    if (!is_float) {
      cleanup();
      return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Input tensor is not float type");
    }
  }

  // Check output data types
  for (size_t i = 0; i < num_outputs; ++i) {
    bool is_float = false;
    RETURN_IF_ERROR(IsFloatTensor(apis.ort_api, static_cast<const OrtValueInfo*>(outputs_data[i]), is_float));
    if (!is_float) {
      cleanup();
      return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Output tensor is not float type");
    }
  }

  cleanup();
  return nullptr;
}

bool SimpleOpBuilder::GetExpectedCounts(const std::string& op_type, size_t& expected_inputs, size_t& expected_outputs) const {
  // Expected input/output counts for simple ops (following QNN provider pattern)
  static const std::unordered_map<std::string, std::pair<size_t, size_t>> op_io_counts = {
    {"Mul", {2, 1}}, {"Add", {2, 1}}, {"Sub", {2, 1}}, {"Div", {2, 1}},
    {"Relu", {1, 1}}, {"Sigmoid", {1, 1}}, {"Tanh", {1, 1}},
    {"Conv", {2, 1}}, {"MatMul", {2, 1}}, {"Gemm", {3, 1}}
  };

  auto it = op_io_counts.find(op_type);
  if (it != op_io_counts.end()) {
    expected_inputs = it->second.first;
    expected_outputs = it->second.second;
    return true;
  }
  return false;
}

OrtStatus* SimpleOpBuilder::IsOpSupported(const ApiPtrs& apis, const OrtNode* node) const {
  const char* op_type = nullptr;
  RETURN_IF_ERROR(apis.ort_api.Node_GetOperatorType(node, &op_type));

  size_t expected_inputs, expected_outputs;
  if (!GetExpectedCounts(op_type, expected_inputs, expected_outputs)) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Unsupported op type");
  }

  // Validate input/output counts
  RETURN_IF_ERROR(ValidateInputOutputCounts(apis, node, expected_inputs, expected_outputs));

  // Validate data types
  RETURN_IF_ERROR(ValidateFloatTensors(apis, node));

  return nullptr;
}

}  // namespace qnn_ep
