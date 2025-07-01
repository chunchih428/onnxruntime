// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

// This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// that allows QNN EP to built either as a static library or a dynamic shared library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// is built as a static library.

#if BUILD_QNN_EP_STATIC_LIB
// Includes when building QNN EP statically
#ifdef _WIN32
#include <Windows.h>
#include <winmeta.h>
#include "core/platform/tracing.h"
#include "core/platform/windows/logging/etw_sink.h"
#endif

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/capture.h"
#include "core/common/path_string.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/float16.h"
#include "core/framework/run_options.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/basic_types.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_cxx_api.h"
#else
// Includes when building QNN EP as a shared library
#include "core/providers/shared_library/provider_api.h"
#endif

namespace onnxruntime::qnn {

#if BUILD_QNN_EP_STATIC_LIB
using Status = onnxruntime::Status;
using Env = onnxruntime::Env;
using AllocatorPtr = onnxruntime::AllocatorPtr;
using IAllocator = onnxruntime::IAllocator;
using OrtMemoryInfo = onnxruntime::OrtMemoryInfo;
using OrtDevice = onnxruntime::OrtDevice;
using GraphViewer = onnxruntime::GraphViewer;
using Node = onnxruntime::Node;
using NodeUnit = onnxruntime::NodeUnit;
using logging::Logger = onnxruntime::logging::Logger;
using ComputeCapability = onnxruntime::ComputeCapability;
using DataLayout = onnxruntime::DataLayout;
using KernelCreateInfo = onnxruntime::KernelCreateInfo;
using KernelDefBuilder = onnxruntime::KernelDefBuilder;
using OpKernelContext = onnxruntime::OpKernelContext;
using OpKernelInfo = onnxruntime::OpKernelInfo;
using TensorShape = onnxruntime::TensorShape;
using MLDataType = onnxruntime::MLDataType;
using DataTypeImpl = onnxruntime::DataTypeImpl;
using AllocatorCreationInfo = onnxruntime::AllocatorCreationInfo;
using AllocatorFactory = onnxruntime::AllocatorFactory;
using OrtMemType = onnxruntime::OrtMemType;
using SafeInt = onnxruntime::SafeInt;
using PathString = onnxruntime::PathString;
using OrtMemTypeDefault = onnxruntime::OrtMemTypeDefault;
using InlinedVector = onnxruntime::InlinedVector;
using InlinedHashMap = onnxruntime::InlinedHashMap;

constexpr const char* QNN_HTP_SHARED = "QnnHtpShared";
constexpr const char* QNN_CPU = "QnnCpu";

const Env& GetDefaultEnv();
using onnxruntime::make_unique;
using onnxruntime::narrow;
using onnxruntime::gsl::narrow_cast;
using onnxruntime::ToUTF8String;

#define ORT_ENFORCE onnxruntime::ORT_ENFORCE
#define ORT_RETURN_IF onnxruntime::ORT_RETURN_IF
#define ORT_RETURN_IF_ERROR onnxruntime::ORT_RETURN_IF_ERROR
#define ORT_RETURN_IF_NOT onnxruntime::ORT_RETURN_IF_NOT
#define ORT_THROW onnxruntime::ORT_THROW
#define ORT_THROW_IF_ERROR onnxruntime::ORT_THROW_IF_ERROR
#define LOGS onnxruntime::LOGS
#define LOGS_DEFAULT onnxruntime::LOGS_DEFAULT
#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE onnxruntime::ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE

#else

using Status = ProviderStatus;
using Env = ProviderEnv;
using AllocatorPtr = ProviderAllocatorPtr;
using IAllocator = ProviderIAllocator;
using OrtMemoryInfo = ProviderOrtMemoryInfo;
using OrtDevice = ProviderOrtDevice;
using GraphViewer = ProviderGraphViewer;
using Node = ProviderNode;
using NodeUnit = ProviderNodeUnit;
using ComputeCapability = ProviderComputeCapability;
using DataLayout = ProviderDataLayout;
using KernelCreateInfo = ProviderKernelCreateInfo;
using KernelDefBuilder = ProviderKernelDefBuilder;
using OpKernelContext = ProviderOpKernelContext;
using OpKernelInfo = ProviderOpKernelInfo;
using TensorShape = ProviderTensorShape;
using MLDataType = ProviderMLDataType;
using DataTypeImpl = ProviderDataTypeImpl;
using AllocatorCreationInfo = ProviderAllocatorCreationInfo;
using AllocatorFactory = ProviderAllocatorFactory;
using OrtMemType = ProviderOrtMemType;
using SafeInt = ProviderSafeInt;
using PathString = ProviderPathString;
using OrtMemTypeDefault = ProviderOrtMemTypeDefault;
using InlinedVector = ProviderInlinedVector;
using InlinedHashMap = ProviderInlinedHashMap;

constexpr const char* QNN_HTP_SHARED = "QnnHtpShared";
constexpr const char* QNN_CPU = "QnnCpu";

namespace logging {
using Logger = ProviderLogger;
using LoggingManager = ProviderLoggingManager;
} // namespace logging

const Env& GetDefaultEnv();

#define ORT_ENFORCE Provider_ORT_ENFORCE
#define ORT_RETURN_IF Provider_ORT_RETURN_IF
#define ORT_RETURN_IF_ERROR Provider_ORT_RETURN_IF_ERROR
#define ORT_RETURN_IF_NOT Provider_ORT_RETURN_IF_NOT
#define ORT_THROW Provider_ORT_THROW
#define ORT_THROW_IF_ERROR Provider_ORT_THROW_IF_ERROR
#define LOGS Provider_LOGS
#define LOGS_DEFAULT Provider_LOGS_DEFAULT
#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE Provider_ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE

#endif

} // namespace onnxruntime::qnn