#pragma once

#include "test/autoep/library/example_plugin_ep_utils.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>

namespace onnxruntime {
class QnnEpFactory;

// Forward declaration for QnnBackendManager
namespace qnn {
class QnnBackendManager;
}

class QnnEp : public OrtEp, public ApiPtrs {
 public:
  struct Config {
    bool enable_ep_context{false};
    bool share_ep_contexts{false};
    bool enable_vtcm_backup_buffer_sharing{false};
  };

  QnnEp(const QnnEpFactory& factory, const std::string& name,
        const Config& config, const OrtLogger* logger);
  ~QnnEp();

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                  const OrtGraph* graph,
                                                  OrtEpGraphSupportInfo* graph_support_info);

  // Helper functions
  bool GraphHasEpContextNode(const OrtGraph* graph);
  int GenerateMetadefId(const OrtGraph* graph, uint64_t& model_hash);
  std::string MakeMetadefName(const OrtGraph* graph);
  bool EpSharedContextsHasAllGraphs(const OrtGraph* graph);
  void PartitionCtxModel(const OrtGraph* graph, size_t num_nodes_in_graph, 
                        OrtEpGraphSupportInfo* graph_support_info);
  static void GetMainEPCtxNodes(QnnEp* ep, const OrtGraph* graph, std::unordered_set<const OrtNode*>& ep_context_nodes);
  void GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                   const std::string& model_path_string,
                                   std::string& context_model_path);

  const QnnEpFactory& factory_;
  std::string name_;
  Config config_;
  const OrtLogger* logger_;
  bool context_cache_enabled_;
  bool share_ep_contexts_;
  bool enable_vtcm_backup_buffer_sharing_;
  std::string context_node_name_prefix_;
  std::string context_cache_path_cfg_;
  
  // Metadef ID generation state
  mutable std::unordered_map<uint64_t, uint64_t> main_graph_hash_;
  mutable std::unordered_map<uint64_t, int> model_metadef_id_;
  
  // Backend manager for QNN operations
  std::shared_ptr<qnn::QnnBackendManager> qnn_backend_manager_;
};

}
