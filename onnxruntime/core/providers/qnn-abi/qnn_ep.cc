#include "qnn_ep.h"

#include "qnn_ep_factory.h"


#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/shared_context.h"
#include "core/providers/qnn-abi/builder/qnn_backend_manager.h"
#include <unordered_map>
#include <vector>
#include <memory>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace onnxruntime {

QnnEp::QnnEp(const QnnEpFactory& factory, const std::string& name,
           const Config& config, const OrtLogger* logger)
    : OrtEp{},
    ApiPtrs{static_cast<const ApiPtrs&>(factory)},
    factory_{factory},
    name_{name},
    config_{config},
    logger_{logger},
    context_cache_enabled_{config.enable_ep_context},
    share_ep_contexts_{config.share_ep_contexts},
    enable_vtcm_backup_buffer_sharing_{config.enable_vtcm_backup_buffer_sharing},
    context_node_name_prefix_{""},
    context_cache_path_cfg_{""}{
        std::cout << "DEBUG: QnnEp constructor called with name: " << name << std::endl;
        GetName = GetNameImpl;
        GetCapability = GetCapabilityImpl;

        // Initialize the backend manager
        qnn::QnnBackendManagerConfig backend_config;
        backend_config.backend_path = "";  // Default backend path
        backend_config.profiling_file_path = "";
        backend_config.device_id = 0;
        backend_config.soc_model = 0;
        qnn_backend_manager_ = qnn::QnnBackendManager::Create(backend_config);
}
QnnEp::~QnnEp() = default;

const char* ORT_API_CALL QnnEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* qnn_ep = static_cast<const QnnEp*>(this_ptr);
  return qnn_ep->name_.c_str();
}

// Helper function to check if a node is supported by QNN
// This is a mock implementation - in real scenario, this would check against QNN's supported ops
static bool IsNodeSupportedByQnn(QnnEp* ep, const OrtNode* node, const char* op_type) {
    ep;
    node;
    if (op_type == nullptr) {
        return false;
    }

    // Mock implementation: Support common ops that QNN typically supports
    static const std::unordered_set<std::string> supported_ops = {
        "Conv", "ConvTranspose", "MatMul", "Gemm", "Add", "Sub", "Mul", "Div",
        "Relu", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "InstanceNormalization",
        "GlobalAveragePool", "AveragePool", "MaxPool", "Reshape", "Transpose", "Concat",
        "Split", "Slice", "Gather", "Squeeze", "Unsqueeze", "Flatten", "Resize",
        "QuantizeLinear", "DequantizeLinear", "Clip", "LeakyRelu", "Elu", "Gelu"
    };

    std::string op_type_str(op_type);

    // Basic support check
    if (supported_ops.find(op_type_str) == supported_ops.end()) {
        return false;
    }

    // Additional validation could be done here:
    // - Check node inputs/outputs
    // - Check attributes
    // - Check data types
    // - Check shapes

    // For now, just return true for supported ops
    return true;
}

// Helper function to get main EPContext nodes - equivalent to GetMainEPCtxNodes
void QnnEp::GetMainEPCtxNodes(QnnEp* ep, const OrtGraph* graph, std::unordered_set<const OrtNode*>& ep_context_nodes) {
    OrtArrayOfConstObjects* graph_nodes = nullptr;
    if (ep->ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
        return;
    }

    size_t num_nodes = 0;
    if (ep->ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return;
    }

    const void* const* node_data = nullptr;
    if (ep->ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return;
    }

    for (size_t i = 0; i < num_nodes; ++i) {
        const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
        const char* op_type = nullptr;

        if (ep->ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
            if (std::string(op_type) == "EPContext") {
                // Check main_context attribute
                const OrtOpAttr* main_context_attr = nullptr;
                if (ep->ort_api.Node_GetAttributeByName(node, "main_context", &main_context_attr) == nullptr && main_context_attr != nullptr) {
                    int64_t is_main_context = 0;
                    size_t out_size = 0;
                    if (ep->ort_api.ReadOpAttr(main_context_attr, ORT_OP_ATTR_INT, &is_main_context, sizeof(is_main_context), &out_size) == nullptr) {
                        // Check source attribute
                        const OrtOpAttr* source_attr = nullptr;
                        if (ep->ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
                            char source_buffer[256] = {0};
                            size_t source_len = 0;
                            if (ep->ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
                                std::string cache_source(source_buffer, source_len);

                                // Convert to lowercase for comparison
                                std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                                             [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

                                if (is_main_context && (cache_source == "qnnexecutionprovider" || cache_source == "qnn")) {
                                    // Log the found EPContext node
                                    if (ep->logger_ != nullptr) {
                                        const char* node_name = nullptr;
                                        size_t node_id = 0;
                                        ep->ort_api.Node_GetName(node, &node_name);
                                        ep->ort_api.Node_GetId(node, &node_id);

                                        std::string log_message = "EPContext Node found: [1] index: [" + std::to_string(node_id) +
                                                                "] name: [" + (node_name ? node_name : "unknown") + "]";
                                        ep->ort_api.Logger_LogMessage(ep->logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
                                                                ORT_FILE, __LINE__, __FUNCTION__);
                                    }
                                    ep_context_nodes.insert(node);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
}

// Helper function to get context ONNX model file path - equivalent to GetContextOnnxModelFilePath
void QnnEp::GetContextOnnxModelFilePath(const std::string& user_context_cache_path,
                                       const std::string& model_path_string,
                                       std::string& context_model_path) {
    // always try the path set by user first, it's the only way to set it if load model from memory
    if (!user_context_cache_path.empty()) {
        context_model_path = user_context_cache_path;
    } else if (!model_path_string.empty()) {  // model loaded from file
        context_model_path = model_path_string;
    }
}

// Helper function to check if graph has EPContext node - equivalent to qnn::GraphHasEpContextNode
bool QnnEp::GraphHasEpContextNode(const OrtGraph* graph) {
    OrtArrayOfConstObjects* graph_nodes = nullptr;
    if (ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
        return false;
    }

    size_t num_nodes = 0;
    if (ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return false;
    }

    const void* const* node_data = nullptr;
    if (ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return false;
    }

    bool has_ep_context = false;
    for (size_t i = 0; i < num_nodes; ++i) {
        const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
        const char* op_type = nullptr;

        if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
            if (std::string(op_type) == "EPContext") {
                // Check if this EPContext node is from QNN
                OrtArrayOfConstObjects* attributes = nullptr;
                if (ort_api.Node_GetAttributes(node, &attributes) == nullptr) {
                    // TODO: Check the 'source' attribute to see if it's "qnn" or "qnnexecutionprovider"
                    // For now, assume any EPContext node is from QNN
                    has_ep_context = true;
                    ort_api.ReleaseArrayOfConstObjects(attributes);
                    break;
                }
            }
        }
    }

    ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return has_ep_context;
}

// Helper function to generate metadef ID - simplified version of ModelMetadefIdGenerator
int QnnEp::GenerateMetadefId(const OrtGraph* graph, uint64_t& model_hash) {
    // Simple hash based on graph pointer (not ideal but works for demo)
    uint64_t graph_hash = reinterpret_cast<uint64_t>(graph);
    model_hash = graph_hash;

    // Generate unique ID for this model
    auto it = model_metadef_id_.find(model_hash);
    if (it != model_metadef_id_.end()) {
        return ++it->second;
    } else {
        model_metadef_id_[model_hash] = 0;
        return 0;
    }
}

// Helper function to make metadef name - equivalent to the lambda in line 895-899
std::string QnnEp::MakeMetadefName(const OrtGraph* graph) {
    uint64_t model_hash;
    int metadef_id = GenerateMetadefId(graph, model_hash);

    std::string result = "QNN";
    if (!context_node_name_prefix_.empty()) {
        result += context_node_name_prefix_;
    }
    result += "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    return result;
}

// Implementation of EpSharedContextsHasAllGraphs - equivalent to the static function in QNNExecutionProvider
bool QnnEp::EpSharedContextsHasAllGraphs(const OrtGraph* graph) {
    OrtArrayOfConstObjects* graph_nodes = nullptr;
    if (ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
        return false;
    }

    size_t num_nodes = 0;
    if (ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return false;
    }

    const void* const* node_data = nullptr;
    if (ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return false;
    }

    bool all_graphs_found = true;

    for (size_t i = 0; i < num_nodes; ++i) {
        const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
        const char* op_type = nullptr;

        if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
            if (std::string(op_type) == "EPContext") {
                // Check the 'source' attribute to verify it's from QNN
                const OrtOpAttr* source_attr = nullptr;
                if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) == nullptr && source_attr != nullptr) {
                    char source_buffer[256] = {0};
                    size_t source_len = 0;
                    if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len) == nullptr) {
                        std::string cache_source(source_buffer, source_len);

                        // Convert to lowercase for comparison
                        std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                                     [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

                        if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
                            // Get the graph name (node name)
                            const char* node_name = nullptr;
                            if (ort_api.Node_GetName(node, &node_name) == nullptr && node_name != nullptr) {
                                std::string graph_name(node_name);
                                bool has_shared_qnn_model = SharedContext::GetInstance().HasQnnModel(graph_name);
                                if (!has_shared_qnn_model) {
                                    // Log the missing graph (equivalent to LOGS(logger, VERBOSE))
                                    if (logger_ != nullptr) {
                                        std::string log_message = "Graph: " + graph_name + " from EpContext node not found from shared EP contexts.";
                                        ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
                                                                ORT_FILE, __LINE__, __FUNCTION__);
                                    }
                                    all_graphs_found = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return all_graphs_found;
}

// Implementation of PartitionCtxModel - equivalent to the static function in QNNExecutionProvider
void QnnEp::PartitionCtxModel(const OrtGraph* graph, size_t num_nodes_in_graph,
                              OrtEpGraphSupportInfo* graph_support_info) {
    // Get all nodes from the graph
    OrtArrayOfConstObjects* graph_nodes = nullptr;
    if (ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
        return;
    }

    size_t num_nodes = 0;
    if (ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return;
    }

    const void* const* node_data = nullptr;
    if (ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
        ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return;
    }

    std::vector<const OrtNode*> supported_nodes;
    std::vector<std::vector<const OrtNode*>> supported_groups;

    // Iterate through all nodes to find EPContext nodes with QNN source (lines 828-847)
    for (size_t i = 0; i < num_nodes; ++i) {
        const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
        const char* op_type = nullptr;

        if (ort_api.Node_GetOperatorType(node, &op_type) && op_type != nullptr) {
            if (std::string(op_type) == "EPContext") {
                // Check the 'source' attribute to verify it's from QNN (lines 829-830)
                const OrtOpAttr* source_attr = nullptr;
                if (ort_api.Node_GetAttributeByName(node, "source", &source_attr) && source_attr != nullptr) {
                    // Read the source attribute as string
                    char source_buffer[256] = {0};
                    size_t source_len = 0;
                    if (ort_api.ReadOpAttr(source_attr, ORT_OP_ATTR_STRING, source_buffer, sizeof(source_buffer) - 1, &source_len)) {
                        std::string cache_source(source_buffer, source_len);

                        // Convert to lowercase for comparison (lines 832-835)
                        std::transform(cache_source.begin(), cache_source.end(), cache_source.begin(),
                                     [](unsigned char c) { return static_cast<unsigned char>(std::tolower(c)); });

                        // Check if source is QNN (line 837)
                        if (cache_source == "qnnexecutionprovider" || cache_source == "qnn") {
                            // Log the supported node (lines 838-841)
                            const char* node_name = nullptr;
                            size_t node_id = 0;
                            ort_api.Node_GetName(node, &node_name);
                            ort_api.Node_GetId(node, &node_id);

                            // Log the supported node information (equivalent to LOGS(logger, VERBOSE))
                            if (logger_ != nullptr) {
                                std::string log_message = "Node supported: [1] index: [" + std::to_string(node_id) +
                                                        "] name: [" + (node_name ? node_name : "unknown") +
                                                        "] Operator type: [EPContext] index: [" + std::to_string(node_id) + "]";
                                ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
                                                        ORT_FILE, __LINE__, __FUNCTION__);
                            }

                            // Add to supported nodes (line 842)
                            supported_nodes.push_back(node);

                            // Each EPContext node gets its own partition group (lines 844-845)
                            std::vector<const OrtNode*> supported_group{node};
                            supported_groups.emplace_back(std::move(supported_group));
                        }
                    }
                }
            }
        }
    }

    // Create partitions for each supported group (lines 849-857)
    // This is equivalent to std::transform with utils::MakeComputeCapability
    for (const auto& supported_partition : supported_groups) {
        if (!supported_partition.empty()) {
            OrtNodeFusionOptions node_fusion_options = {};
            node_fusion_options.ort_version_supported = ORT_API_VERSION;
            node_fusion_options.drop_constant_initializers = false;  // EPContext nodes don't drop initializers (line 856)

            // Add this partition to the graph support info
            // This is the EP API equivalent of utils::MakeComputeCapability
            ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                    supported_partition.data(),
                                                    supported_partition.size(),
                                                    &node_fusion_options);
        }
    }

    // Log summary (lines 859-863)
    const size_t num_of_partitions = supported_groups.size();
    if (logger_ != nullptr) {
        std::string summary_msg = "Number of partitions supported by QNN EP: " + std::to_string(num_of_partitions) +
                                ", number of nodes in the graph: " + std::to_string(num_nodes_in_graph) +
                                ", number of nodes supported by QNN: " + std::to_string(num_of_partitions);
        ort_api.Logger_LogMessage(logger_, ORT_LOGGING_LEVEL_INFO, summary_msg.c_str(),
                                ORT_FILE, __LINE__, __FUNCTION__);
    }

    ort_api.ReleaseArrayOfConstObjects(graph_nodes);
}


OrtStatus* ORT_API_CALL QnnEp::GetCapabilityImpl(OrtEp* this_ptr,
                                                const OrtGraph* graph,
                                                OrtEpGraphSupportInfo* graph_support_info) {
    QnnEp* ep = static_cast<QnnEp*>(this_ptr);

    // Check if this is a subgraph - similar to graph_viewer.IsSubgraph()
    const OrtNode* parent_node = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Graph_GetParentNode(graph, &parent_node));
    if (parent_node != nullptr) {
        // This is a subgraph, return empty result
        return nullptr;
    }

    // Get number of nodes in graph - similar to graph_viewer.NumberOfNodes()
    OrtArrayOfConstObjects* graph_nodes = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Graph_GetNodes(graph, &graph_nodes));

    size_t num_nodes_in_graph = 0;
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes_in_graph));

    // Early exit if no nodes
    if (num_nodes_in_graph == 0) {
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return nullptr;
    }

    // Implementation of lines 892-899 from QNNExecutionProvider::GetCapability
    // const auto& logger = *GetLogger();  // We use ep->logger_ instead
    bool is_qnn_ctx_model = ep->GraphHasEpContextNode(graph);

    // This is the lambda function gen_metadef_name from lines 895-899
    // const auto gen_metadef_name = [&]() {
    //   uint64_t model_hash;
    //   int metadef_id = metadef_id_generator_->GenerateId(graph_viewer, model_hash);
    //   return MakeString(QNN, context_node_name_prefix_, "_", model_hash, "_", metadef_id);
    // };
    auto gen_metadef_name = [ep, graph]() -> std::string {
        return ep->MakeMetadefName(graph);
    };

    // Get graph inputs and outputs for context
    OrtArrayOfConstObjects* graph_inputs = nullptr;
    OrtArrayOfConstObjects* graph_outputs = nullptr;
    RETURN_IF_ERROR(ep->ort_api.Graph_GetInputs(graph, &graph_inputs));
    RETURN_IF_ERROR(ep->ort_api.Graph_GetOutputs(graph, &graph_outputs));

    // Check for EP context sharing (from lines 901-910)
    if (is_qnn_ctx_model && ep->config_.share_ep_contexts && SharedContext::GetInstance().HasSharedQnnModels()) {
        if (ep->EpSharedContextsHasAllGraphs(graph)) {
            // Call PartitionCtxModel and return early (line 907-908)
            ep->PartitionCtxModel(graph, num_nodes_in_graph, graph_support_info);
            ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
            ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
            ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
            return nullptr;
        }
    }

    // Implementation of lines 913-934: VTCM backup buffer sharing
    std::unordered_map<std::string, std::unique_ptr<std::vector<std::string>>> context_bin_map;
    if (ep->enable_vtcm_backup_buffer_sharing_) {
        std::unordered_set<const OrtNode*> ep_ctx_nodes;
        GetMainEPCtxNodes(ep, graph, ep_ctx_nodes);

        // Get the graph model path - we need to mock this since EP API doesn't have direct access
        std::string model_path_string = "";  // In real implementation, this would come from graph metadata
        std::string context_model_path;
        ep->GetContextOnnxModelFilePath(ep->context_cache_path_cfg_, model_path_string, context_model_path);

        std::filesystem::path parent_path = std::filesystem::path(context_model_path).parent_path();

        for (auto& ep_ctx_node : ep_ctx_nodes) {
            // Get the ep_cache_context attribute from the node
            const OrtOpAttr* ep_cache_context_attr = nullptr;
            if (ep->ort_api.Node_GetAttributeByName(ep_ctx_node, "ep_cache_context", &ep_cache_context_attr) == nullptr && ep_cache_context_attr != nullptr) {
                char context_buffer[512] = {0};
                size_t context_len = 0;
                if (ep->ort_api.ReadOpAttr(ep_cache_context_attr, ORT_OP_ATTR_STRING, context_buffer, sizeof(context_buffer) - 1, &context_len) == nullptr) {
                    std::string context_bin_filepath(parent_path.string());
                    context_bin_filepath.append("/").append(std::string(context_buffer, context_len));

                    if (context_bin_map.find(context_bin_filepath) == context_bin_map.end()) {
                        context_bin_map.emplace(context_bin_filepath, std::make_unique<std::vector<std::string>>());
                        // Push context bin filepath for lookup between sessions
                        context_bin_map.at(context_bin_filepath)->push_back(context_bin_filepath);
                    }

                    // Add the node name to the context bin map
                    const char* node_name = nullptr;
                    if (ep->ort_api.Node_GetName(ep_ctx_node, &node_name) == nullptr && node_name != nullptr) {
                        context_bin_map.at(context_bin_filepath)->push_back(std::string(node_name));
                    }
                }
            }
        }
    }

    // Implementation of lines 939-943: SetupBackend call
    OrtStatus* rt = ep->qnn_backend_manager_->SetupBackend(ep->logger_, is_qnn_ctx_model,
                                                           ep->context_cache_enabled_ && false,  // enable_spill_fill_buffer_ (not implemented)
                                                           ep->share_ep_contexts_,
                                                           ep->enable_vtcm_backup_buffer_sharing_,
                                                           context_bin_map);

    context_bin_map.clear();

    if (rt != nullptr) {
        // Log error message equivalent to lines 947-949
        if (ep->logger_ != nullptr) {
            const char* error_msg = ep->ort_api.GetErrorMessage(rt);
            std::string full_error_msg = "QNN SetupBackend failed: " + std::string(error_msg ? error_msg : "Unknown error");
            ep->ort_api.Logger_LogMessage(ep->logger_, ORT_LOGGING_LEVEL_ERROR, full_error_msg.c_str(),
                                        ORT_FILE, __LINE__, __FUNCTION__);
        }
        // Clean up and return the error status (equivalent to line 949)
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
        ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
        return rt;
    }

    // For context models, handle differently (from lines 969-973)
    if (is_qnn_ctx_model) {
        // Call PartitionCtxModel equivalent - lines 971 in original
        ep->PartitionCtxModel(graph, num_nodes_in_graph, graph_support_info);
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
        ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);
        return nullptr;
    }

    // Get node data for processing
    const void* const* node_data = nullptr;
    RETURN_IF_ERROR(ep->ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data));

    // Mock: Get QDQ node units (simplified)
    // TODO: Implement GetQDQNodeUnits equivalent using EP API

    // Analyze nodes for QNN support
    std::vector<const OrtNode*> supported_nodes;

    for (size_t i = 0; i < num_nodes_in_graph; ++i) {
        const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);

        // Get node information
        const char* op_type = nullptr;
        const char* node_name = nullptr;
        size_t node_id = 0;

        RETURN_IF_ERROR(ep->ort_api.Node_GetOperatorType(node, &op_type));
        RETURN_IF_ERROR(ep->ort_api.Node_GetName(node, &node_name));
        RETURN_IF_ERROR(ep->ort_api.Node_GetId(node, &node_id));

        // Mock: Check if node is supported by QNN
        bool is_supported = IsNodeSupportedByQnn(ep, node, op_type);

        if (is_supported) {
            supported_nodes.push_back(node);
        }
    }

    // Clean up intermediate resources
    ep->ort_api.ReleaseArrayOfConstObjects(graph_inputs);
    ep->ort_api.ReleaseArrayOfConstObjects(graph_outputs);

    // If no supported nodes, return empty
    if (supported_nodes.empty()) {
        ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);
        return nullptr;
    }

    // Mock: Validate partitions (simplified)
    // TODO: Implement partition validation similar to original

    // Create partitions from supported nodes - equivalent to utils::CreateSupportedPartitions (line 1013-1014)
    if (!supported_nodes.empty()) {
        // Mock: Validate partitions (filter out single QuantizeLinear/DequantizeLinear nodes)
        std::vector<const OrtNode*> valid_supported_nodes;
        for (const OrtNode* node : supported_nodes) {
            const char* op_type = nullptr;
            if (ep->ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
                std::string op_type_str(op_type);
                // For single node partitions, skip QuantizeLinear/DequantizeLinear
                if (supported_nodes.size() == 1 &&
                    (op_type_str == "QuantizeLinear" || op_type_str == "DequantizeLinear")) {
                    continue;
                }
                valid_supported_nodes.push_back(node);
            }
        }

        if (!valid_supported_nodes.empty()) {
            OrtNodeFusionOptions node_fusion_options = {};
            node_fusion_options.ort_version_supported = ORT_API_VERSION;
            node_fusion_options.drop_constant_initializers = true;

            // Use the generated metadef name for the fused node
            // The gen_metadef_name lambda would be used in the actual partition creation
            // For now, we use the EP API to add nodes to fusion

            RETURN_IF_ERROR(ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                                                                          valid_supported_nodes.data(),
                                                                          valid_supported_nodes.size(),
                                                                          &node_fusion_options));
        }
    }

    // Clean up
    ep->ort_api.ReleaseArrayOfConstObjects(graph_nodes);

    return nullptr;
}



}
