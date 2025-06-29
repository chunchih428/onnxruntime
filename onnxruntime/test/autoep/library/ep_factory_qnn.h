// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ep_data_transfer.h"
#include "example_plugin_ep_utils.h"

/// <summary>
/// QNN EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class QnnEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  QnnEpFactory(const char* ep_name, ApiPtrs apis, OrtHardwareDeviceType device_type = OrtHardwareDeviceType_CPU, const char* backend_type = "cpu");

  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

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