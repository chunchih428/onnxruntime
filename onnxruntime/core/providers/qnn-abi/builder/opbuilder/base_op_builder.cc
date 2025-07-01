// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/opbuilder/base_op_builder.h"
#include <unordered_map>
#include <utility>

namespace qnn_ep {

std::string BaseOpBuilder::GetOpBuilderType() const {
  return op_builder_type_;
}

OrtStatus* BaseOpBuilder::IsOpSupported(const ApiPtrs& apis, const OrtNode* node) const {
  return AddToModelBuilder(apis, node, nullptr);
}

OrtStatus* BaseOpBuilder::AddToModelBuilder(const ApiPtrs& apis, const OrtNode* node,
                                            void* qnn_model_wrapper) const {
  OrtStatus* status = ProcessInputs(apis, node, qnn_model_wrapper);
  if (status != nullptr) {
    return status;
  }
  
  return ProcessAttributesAndOutputs(apis, node, qnn_model_wrapper);
}

OrtStatus* BaseOpBuilder::ValidateInputOutputCounts(const ApiPtrs& apis, const OrtNode* node,
                                                    size_t expected_inputs, size_t expected_outputs) const {
  size_t input_count = 0;
  size_t output_count = 0;
  
  apis.ort_api.Node_GetInputCount(node, &input_count);
  apis.ort_api.Node_GetOutputCount(node, &output_count);

  if (input_count != expected_inputs) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input count");
  }

  if (output_count != expected_outputs) {
    return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output count");
  }

  return nullptr;
}

OrtStatus* BaseOpBuilder::ValidateFloatTensors(const ApiPtrs& apis, const OrtNode* node) const {
  size_t input_count = 0;
  size_t output_count = 0;
  
  apis.ort_api.Node_GetInputCount(node, &input_count);
  apis.ort_api.Node_GetOutputCount(node, &output_count);

  // Check input types
  for (size_t i = 0; i < input_count; ++i) {
    const OrtTypeInfo* type_info = nullptr;
    apis.ort_api.Node_GetInputTypeInfo(node, i, &type_info);
    
    ONNXTensorElementDataType element_type;
    apis.ort_api.GetTensorElementType(type_info, &element_type);
    
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Expected float tensor");
    }
  }

  // Check output types
  for (size_t i = 0; i < output_count; ++i) {
    const OrtTypeInfo* type_info = nullptr;
    apis.ort_api.Node_GetOutputTypeInfo(node, i, &type_info);
    
    ONNXTensorElementDataType element_type;
    apis.ort_api.GetTensorElementType(type_info, &element_type);
    
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return apis.ort_api.CreateStatus(ORT_INVALID_ARGUMENT, "Expected float tensor");
    }
  }

  return nullptr;
}

OrtStatus* BaseOpBuilder::ProcessInputs(const ApiPtrs& apis, const OrtNode* node,
                                        void* qnn_model_wrapper) const {
  return nullptr;  // OK by default
}

OrtStatus* BaseOpBuilder::ProcessAttributesAndOutputs(const ApiPtrs& apis, const OrtNode* node,
                                                      void* qnn_model_wrapper) const {
  return ProcessOutputs(apis, node, qnn_model_wrapper);
}

OrtStatus* BaseOpBuilder::ProcessOutputs(const ApiPtrs& apis, const OrtNode* node,
                                         void* qnn_model_wrapper) const {
  return nullptr;  // OK by default
}

const std::string& BaseOpBuilder::GetQnnOpType(const std::string& onnx_op_type) {
  static const std::unordered_map<std::string, std::string> onnx_op_type_to_qnn_op_type = {
      {"Add", "ElementWiseAdd"},
      {"Mul", "ElementWiseMultiply"},
      {"Abs", "ElementWiseAbs"},
      {"And", "ElementWiseAnd"},
      {"Asin", "ElementWiseAsin"},
      {"Atan", "ElementWiseAtan"},
      {"Ceil", "ElementWiseCeil"},
      {"Sign", "ElementWiseSign"},
      {"Cast", "Cast"},
      {"Clip", "ReluMinMax"},
      {"Cos", "ElementWiseCos"},
      {"Div", "ElementWiseDivide"},
      {"Equal", "ElementWiseEqual"},
      {"Exp", "ElementWiseExp"},
      {"Floor", "ElementWiseFloor"},
      {"Gather", "Gather"},
      {"GatherElements", "GatherElements"},
      {"Greater", "ElementWiseGreater"},
      {"GreaterOrEqual", "ElementWiseGreaterEqual"},
      {"Less", "ElementWiseLess"},
      {"LessOrEqual", "ElementWiseLessEqual"},
      {"Log", "ElementWiseLog"},
      {"LSTM", "LSTM"},
      {"Max", "ElementWiseMaximum"},
      {"Min", "ElementWiseMinimum"},
      {"Neg", "ElementWiseNeg"},
      {"Not", "ElementWiseNot"},
      {"Or", "ElementWiseOr"},
      {"Pow", "ElementWisePower"},
      {"PRelu", "Prelu"},
      {"LeakyRelu", "Prelu"},
      {"ReduceMax", "ReduceMax"},
      {"ReduceMean", "ReduceMean"},
      {"ReduceMin", "ReduceMin"},
      {"ReduceProd", "ReduceProd"},
      {"ReduceSum", "ReduceSum"},
      {"Round", "ElementWiseRound"},
      {"Where", "ElementWiseSelect"},
      {"ScatterND", "ScatterND"},
      {"Sigmoid", "Sigmoid"},
      {"Sin", "ElementWiseSin"},
      {"Slice", "StridedSlice"},
      {"Split", "Split"},
      {"Softmax", "Softmax"},
      {"Sqrt", "ElementWiseSquareRoot"},
      {"Sub", "ElementWiseSubtract"},
      {"Sum", "ElementWiseAdd"},
      {"Tanh", "Tanh"},
      {"Transpose", "Transpose"},
      {"GridSample", "GridSample"},
      {"LpNormalization", "L2Norm"},
      {"DequantizeLinear", "Dequantize"},
      {"QuantizeLinear", "Quantize"},
      {"MatMul", "MatMul"},
      {"Elu", "Elu"},
      {"Relu", "Relu"},
      {"Gelu", "Gelu"},
      {"HardSigmoid", "ElementWiseNeuron"},
      {"HardSwish", "HardSwish"},
      {"DepthToSpace", "DepthToSpace"},
      {"SpaceToDepth", "SpaceToDepth"},
      {"Conv", "Conv2d"},
      {"ConvTranspose", "TransposeConv2d"},
      {"GlobalAveragePool", "PoolAvg2d"},
      {"AveragePool", "PoolAvg2d"},
      {"MaxPool", "PoolMax2d"},
      {"GlobalMaxPool", "PoolMax2d"},
      {"Reshape", "Reshape"},
      {"Resize", "Resize"},
      {"Upsample", "Resize"},
      {"Flatten", "Reshape"},
      {"Squeeze", "Reshape"},
      {"Unsqueeze", "Reshape"},
      {"LogSoftmax", "LogSoftmax"},
      {"Concat", "Concat"},
      {"CumSum", "CumulativeSum"},
      {"Gemm", "FullyConnected"},
      {"ArgMax", "ArgMax"},
      {"ArgMin", "ArgMin"},
      {"Tile", "Tile"},
      {"TopK", "TopK"},
      {"InstanceNormalization", "InstanceNorm"},
      {"BatchNormalization", "Batchnorm"},
      {"LayerNormalization", "LayerNorm"},
      {"LRN", "LRN"},
      {"Pad", "Pad"},
      {"Expand", "ElementWiseMultiply"}
  };
  
  auto it = onnx_op_type_to_qnn_op_type.find(onnx_op_type);
  if (it != onnx_op_type_to_qnn_op_type.end()) {
    return it->second;
  }
  
  // Return the original type if not found in mapping
  static std::string fallback = onnx_op_type;
  fallback = onnx_op_type;  // Update fallback with current type
  return fallback;
}

}  // namespace qnn_ep