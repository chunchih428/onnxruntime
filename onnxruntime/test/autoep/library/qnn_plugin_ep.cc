// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep_factory_qnn.h"

// To make symbols visible on macOS/iOS and Windows
#ifdef __APPLE__
#define EXPORT_SYMBOL __attribute__((visibility("default")))
#elif defined(_WIN32)
#define EXPORT_SYMBOL __declspec(dllexport)
#else
#define EXPORT_SYMBOL
#endif

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Factory could use registration_name or define its own EP name.
  // Create factory for CPU backend by default
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<QnnEpFactory>(registration_name,
                                                                             ApiPtrs{*ort_api, *ep_api,
                                                                                     *model_editor_api},
                                                                             OrtHardwareDeviceType_CPU,
                                                                             "cpu");

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<QnnEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
