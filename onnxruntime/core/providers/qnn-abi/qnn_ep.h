// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>
#include <memory>

#include "test/autoep/library/example_plugin_ep_utils.h"
#include "core/providers/qnn-abi/rpcmem_library.h"
#include "core/providers/qnn-abi/shared_context.h"

class QnnEpFactory;
struct QnnMulKernel;

/// <summary>
/// QNN EP following QNN provider pattern with builder support
/// </summary>
class QnnEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context = false;
    bool enable_htp_shared_memory_allocator = false;
    bool share_ep_contexts = false;
    // Other EP configs (typically extracted from OrtSessionOptions or OrtHardwareDevice(s))
  };

  QnnEp(QnnEpFactory& factory, const std::string& name, const Config& config, const OrtLogger& logger);

  ~QnnEp();

  std::unordered_map<std::string, std::unique_ptr<QnnMulKernel>>& Kernels() {
    return kernels_;
  }

  // Helper methods for RpcMem and SharedContext
  bool IsHtpSharedMemoryAllocatorAvailable() const { return rpcmem_library_ != nullptr; }
  
  // Note: This method signature may need adjustment based on the actual AllocatorPtr definition in your codebase
  // std::vector<AllocatorPtr> CreatePreferredAllocators();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                          OrtEpGraphSupportInfo* graph_support_info);
  static OrtStatus* ORT_API_CALL CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                           _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                           _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                           _Out_writes_(count) OrtNode** ep_context_nodes);
  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                       OrtNodeComputeInfo** node_compute_infos,
                                                       size_t num_node_compute_infos);

  OrtStatus* CreateEpContextNodes(gsl::span<const OrtNode*> fused_nodes,
                                  /*out*/ gsl::span<OrtNode*> ep_context_nodes);

  OrtStatus* QnnEp::SaveConstantInitializers(const OrtGraph* graph);

  // QNN EP pattern methods
  std::vector<const OrtNode*> GetSupportedNodes(const OrtGraph* graph) const;

  QnnEpFactory& factory_;
  std::string name_;
  Config config_{};
  const OrtLogger& logger_;
  std::unordered_map<std::string, std::unique_ptr<QnnMulKernel>> kernels_;
  std::unordered_map<std::string, FloatInitializer> float_initializers_;
  
  // RpcMem support for HTP shared memory allocator
  std::shared_ptr<qnn::RpcMemLibrary> rpcmem_library_;
  
  // SharedContext usage
  bool context_cache_enabled_ = false;
  bool share_ep_contexts_ = false;
};
