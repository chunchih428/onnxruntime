// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cassert>

#include "qnn_ep.h"
#include "test/autoep/library/ep_allocator.h"
#include "test/autoep/library/ep_data_transfer.h"
#include "test/autoep/library/example_plugin_ep_utils.h"

// QNN memory type constants (from QNN provider)
static constexpr const char* QNN_CPU = "QnnCpu";
static constexpr const char* QNN_HTP = "QnnHtp";

// Qualcomm vendor ID (from QNN provider)
static constexpr uint32_t QUALCOMM_VENDOR_ID = 'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24);

/// <summary>
/// QNN EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class QnnEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  QnnEpFactory::QnnEpFactory(const char* ep_name, ApiPtrs apis, OrtHardwareDeviceType device_type, const char* backend_type)
    : ApiPtrs(apis), ep_name_{ep_name}, ort_hw_device_type_{device_type}, qnn_backend_type_{backend_type} {
    ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;

    GetSupportedDevices = GetSupportedDevicesImpl;

    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;

    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;

    CreateDataTransfer = CreateDataTransferImpl;

    // for the sake of this example we specify a CPU allocator with no arena and 1K alignment (arbitrary)
    // as well as GPU and GPU shared memory. the actual EP implementation would typically define two at most for a
    // device (one for device memory and one for shared memory for data transfer between device and CPU)

    // Setup CPU memory info for QNN EP
    OrtMemoryInfo* mem_info = nullptr;
    auto* status = ort_api.CreateMemoryInfo_V2(QNN_CPU, OrtMemoryInfoDeviceType_CPU,
                                              QUALCOMM_VENDOR_ID, /* device_id */ 0,
                                              OrtDeviceMemoryType_DEFAULT,
                                              /*alignment*/ 1024,
                                              OrtAllocatorType::OrtDeviceAllocator,
                                              &mem_info);
    assert(status == nullptr);
    cpu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);

    // Setup NPU memory info for QNN EP
    mem_info = nullptr;
    status = ort_api.CreateMemoryInfo_V2(QNN_HTP, OrtMemoryInfoDeviceType_NPU,
                                        QUALCOMM_VENDOR_ID, /* device_id */ 0,
                                        OrtDeviceMemoryType_DEFAULT,
                                        /*alignment*/ 0,
                                        OrtAllocatorType::OrtDeviceAllocator,
                                        &mem_info);
    assert(status == nullptr);
    npu_memory_info_ = MemoryInfoUniquePtr(mem_info, ort_api.ReleaseMemoryInfo);
  }


  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

 private:
  /*static*/
  const char* ORT_API_CALL QnnEpFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->ep_name_.c_str();
  }

  /*static*/
  const char* ORT_API_CALL QnnEpFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const QnnEpFactory*>(this_ptr);
    return factory->vendor_.c_str();
  }

  /*static*/
  OrtStatus* ORT_API_CALL QnnEpFactory::GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                                  const OrtHardwareDevice* const* devices,
                                                                  size_t num_devices,
                                                                  OrtEpDevice** ep_devices,
                                                                  size_t max_ep_devices,
                                                                  size_t* p_num_ep_devices) noexcept {
    size_t& num_ep_devices = *p_num_ep_devices;
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);

    for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
      // C API
      const OrtHardwareDevice& device = *devices[i];
      // Match device type only (like example EP) - don't require specific vendor ID for testing
      if (factory->ort_api.HardwareDevice_Type(&device) == factory->ort_hw_device_type_) {
        // Create metadata and options following QNN provider pattern
        OrtKeyValuePairs* ep_metadata = nullptr;
        OrtKeyValuePairs* ep_options = nullptr;
        factory->ort_api.CreateKeyValuePairs(&ep_metadata);
        factory->ort_api.CreateKeyValuePairs(&ep_options);

        // QNN EP metadata and options to match test expectations
        factory->ort_api.AddKeyValuePair(ep_metadata, "version", "1.0");
        factory->ort_api.AddKeyValuePair(ep_metadata, "backend_type", factory->qnn_backend_type_.c_str());
        factory->ort_api.AddKeyValuePair(ep_options, "enable_htp_fp16_precision", "false");
        factory->ort_api.AddKeyValuePair(ep_options, "qnn_context_priority", "normal");
        // Add RpcMem and SharedContext options
        factory->ort_api.AddKeyValuePair(ep_options, "enable_htp_shared_memory_allocator", "true");
        factory->ort_api.AddKeyValuePair(ep_options, "share_ep_contexts", "false");

        // OrtEpDevice copies ep_metadata and ep_options.
        OrtEpDevice* ep_device = nullptr;
        auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options,
                                                                  &ep_device);

        factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
        factory->ort_api.ReleaseKeyValuePairs(ep_options);

        if (status != nullptr) {
          return status;
        }

        // Register allocator info based on device type
        if (factory->ort_hw_device_type_ == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
          RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->cpu_memory_info_.get()));
        } else if (factory->ort_hw_device_type_ == OrtHardwareDeviceType::OrtHardwareDeviceType_NPU) {
          // Register NPU device memory
          RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->npu_memory_info_.get()));
          // Register HTP shared memory for data transfer between NPU and CPU
          RETURN_IF_ERROR(factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, factory->htp_shared_memory_info_.get()));
        }

        ep_devices[num_ep_devices++] = ep_device;
      }

      // C++ API equivalent. Throws on error.
      //{
      //  Ort::ConstHardwareDevice device(devices[i]);
      //  if (device.Type() == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      //    Ort::KeyValuePairs ep_metadata;
      //    Ort::KeyValuePairs ep_options;
      //    ep_metadata.Add("version", "0.1");
      //    ep_options.Add("run_really_fast", "true");
      //    Ort::EpDevice ep_device{*this_ptr, device, ep_metadata.GetConst(), ep_options.GetConst()};
      //    ep_devices[num_ep_devices++] = ep_device.release();
      //  }
      //}
    }

    return nullptr;
  }

  /*static*/
  OrtStatus* ORT_API_CALL QnnEpFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                        const OrtHardwareDevice* const* /*devices*/,
                                                        const OrtKeyValuePairs* const* /*ep_metadata*/,
                                                        size_t num_devices,
                                                        const OrtSessionOptions* session_options,
                                                        const OrtLogger* logger,
                                                        OrtEp** ep) noexcept {
    auto* factory = static_cast<QnnEpFactory*>(this_ptr);
    *ep = nullptr;

    if (num_devices != 1) {
      // we support CPU and NPU devices, but only one at a time
      // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
      // the EP has been selected for.
      return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "QNN EP only supports selection for one device at a time.");
    }

    // Create the execution provider
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                      OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                      "Creating QNN EP", ORT_FILE, __LINE__, __FUNCTION__));

    // use properties from the device and ep_metadata if needed
    // const OrtHardwareDevice* device = devices[0];
    // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

    // Create EP configuration from session options, if needed.
    // Note: should not store a direct reference to the session options object as its lifespan is not guaranteed.
    std::string ep_context_enable;
    std::string htp_shared_memory_enable;
    std::string share_ep_contexts_enable;

    RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(factory->ort_api, *session_options,
                                                  "ep.context_enable", "0", ep_context_enable));
    RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(factory->ort_api, *session_options,
                                                  "ep.enable_htp_shared_memory_allocator", "0", htp_shared_memory_enable));
    RETURN_IF_ERROR(GetSessionConfigEntryOrDefault(factory->ort_api, *session_options,
                                                  "ep.share_ep_contexts", "0", share_ep_contexts_enable));

    QnnEp::Config config = {};
    config.enable_ep_context = ep_context_enable == "1";
    config.enable_htp_shared_memory_allocator = htp_shared_memory_enable == "1";
    config.share_ep_contexts = share_ep_contexts_enable == "1";

    auto qnn_ep = std::make_unique<QnnEp>(*factory, factory->ep_name_, config, *logger);

    *ep = qnn_ep.release();
    return nullptr;
  }

  /*static*/
  void ORT_API_CALL QnnEpFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    QnnEp* qnn_ep = static_cast<QnnEp*>(ep);
    delete qnn_ep;
  }

  /*static*/
  OrtStatus* ORT_API_CALL QnnEpFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                                const OrtMemoryInfo* memory_info,
                                                                const OrtKeyValuePairs* /*allocator_options*/,
                                                                OrtAllocator** allocator) noexcept {
    auto& factory = *static_cast<QnnEpFactory*>(this_ptr);
    *allocator = nullptr;

    // NOTE: The factory implementation can return a shared OrtAllocator* instead of creating a new instance on each call.
    //       To do this just make ReleaseAllocatorImpl a no-op.

    // NOTE: If OrtMemoryInfo has allocator type (call MemoryInfoGetType) of OrtArenaAllocator, an ORT BFCArena
    //       will be added to wrap the returned OrtAllocator. The EP is free to implement its own arena, and if it
    //       wants to do this the OrtMemoryInfo MUST be created with an allocator type of OrtDeviceAllocator.

    // Match memory info to create appropriate allocator
    if (memory_info == factory.cpu_memory_info_.get()) {
      // Create CPU allocator
      auto cpu_allocator = std::make_unique<CustomAllocator>(memory_info);
      *allocator = cpu_allocator.release();
    } else if (memory_info == factory.npu_memory_info_.get()) {
      // Create NPU device memory allocator
      auto npu_allocator = std::make_unique<CustomAllocator>(memory_info);
      *allocator = npu_allocator.release();
    } else if (memory_info == factory.htp_shared_memory_info_.get()) {
      // Create HTP shared memory allocator for data transfer
      // This would typically use the QNN provider's HtpSharedMemoryAllocator pattern
      auto htp_shared_allocator = std::make_unique<CustomAllocator>(memory_info);
      *allocator = htp_shared_allocator.release();
    } else if (memory_info == factory.default_gpu_memory_info_.get()) {
      // Create GPU allocator (future use)
      return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "QNN EP GPU support not yet implemented.");
    } else if (memory_info == factory.host_accessible_gpu_memory_info_.get()) {
      // Create GPU pinned/shared memory allocator (future use)
      return factory.ort_api.CreateStatus(ORT_NOT_IMPLEMENTED, "QNN EP GPU shared memory not yet implemented.");
    } else {
      return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                          "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                          "Value did not come directly from an OrtEpDevice returned by this factory.");
    }

    return nullptr;
  }

  /*static*/
  void ORT_API_CALL QnnEpFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
    delete static_cast<CustomAllocator*>(allocator);
  }

  /*static*/
  OrtStatus* ORT_API_CALL QnnEpFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                                  OrtDataTransferImpl** data_transfer) noexcept {
    auto& factory = *static_cast<QnnEpFactory*>(this_ptr);
    *data_transfer = factory.data_transfer_impl_.get();

    return nullptr;
  }


  const std::string ep_name_;            // EP name
  const std::string vendor_{"Qualcomm"};  // EP vendor name

  // Qualcomm vendor ID (same as QNN provider)
  const uint32_t vendor_id_{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  const OrtHardwareDeviceType ort_hw_device_type_;  // Supported hardware device type
  const std::string qnn_backend_type_;              // QNN backend type

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr cpu_memory_info_;

  // NPU/HTP memory info for QNN backend
  MemoryInfoUniquePtr npu_memory_info_;
  MemoryInfoUniquePtr htp_shared_memory_info_;

  // GPU memory info (if needed in future)
  MemoryInfoUniquePtr default_gpu_memory_info_;
  MemoryInfoUniquePtr host_accessible_gpu_memory_info_;

  std::unique_ptr<ExampleDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};
