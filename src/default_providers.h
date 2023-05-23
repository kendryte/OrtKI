#pragma once
#include <memory>
#include <optional>
#include <onnxruntime/core/framework/execution_provider.h>
#include <onnxruntime/core/providers/providers.h>

namespace ortki
{
// unique_ptr providers with default values for session registration
    std::unique_ptr<onnxruntime::IExecutionProvider> DefaultCpuExecutionProvider(bool enable_arena = true);
}
