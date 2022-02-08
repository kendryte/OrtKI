#pragma once
#include <memory>
#include <optional>
#include "core/common/optional.h"
#include "core/providers/providers.h"
#include "core/framework/execution_provider.h"

using namespace onnxruntime;
namespace ort_ki
{
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ACL(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_ArmNN(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CoreML(uint32_t);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptionsV2* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Dnnl(int use_arena);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MIGraphX(const OrtMIGraphXProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nnapi(
        uint32_t flags, const std::optional<std::string>& partitioning_stop_ops_list);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Nuphar(bool, const char*);
//std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Stvm(const char*);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
        const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads, bool use_compiled_network, const char* blob_dump_path);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(const OrtOpenVINOProviderOptions* params);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rknpu();
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Rocm(const OrtROCMProviderOptions* provider_options);
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Tensorrt(const OrtTensorRTProviderOptions* params);

// EP for internal testing
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_InternalTesting(const std::unordered_set<std::string>& supported_ops);


// unique_ptr providers with default values for session registration
    std::unique_ptr<IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);

    std::unique_ptr<IExecutionProvider> DefaultCudaExecutionProvider();

    std::unique_ptr<IExecutionProvider> DefaultDnnlExecutionProvider(bool enable_arena = true);

    std::unique_ptr<IExecutionProvider> DefaultNupharExecutionProvider(bool allow_unaligned_buffers = true);

//std::unique_ptr<IExecutionProvider> DefaultStvmExecutionProvider();
    std::unique_ptr<IExecutionProvider> DefaultTensorrtExecutionProvider();

    std::unique_ptr<IExecutionProvider> TensorrtExecutionProviderWithOptions(const OrtTensorRTProviderOptions *params);

    std::unique_ptr<IExecutionProvider> DefaultMIGraphXExecutionProvider();

    std::unique_ptr<IExecutionProvider> MIGraphXExecutionProviderWithOptions(const OrtMIGraphXProviderOptions *params);

    std::unique_ptr<IExecutionProvider> DefaultOpenVINOExecutionProvider();

    std::unique_ptr<IExecutionProvider> DefaultNnapiExecutionProvider();

    std::unique_ptr<IExecutionProvider> DefaultRknpuExecutionProvider();

    std::unique_ptr<IExecutionProvider> DefaultAclExecutionProvider(bool enable_arena = true);

    std::unique_ptr<IExecutionProvider> DefaultArmNNExecutionProvider(bool enable_arena = true);

    std::unique_ptr<IExecutionProvider> DefaultRocmExecutionProvider();

    std::unique_ptr<IExecutionProvider> DefaultCoreMLExecutionProvider();

// EP for internal testing
    std::unique_ptr<IExecutionProvider> DefaultInternalTestingExecutionProvider(
            const std::unordered_set<std::string> &supported_ops);
}