#include <memory>
#include "default_providers.h"
#include <onnxruntime/core/providers/cpu/cpu_provider_factory_creator.h>

using namespace onnxruntime;
using namespace ortki;

std::unique_ptr<IExecutionProvider> ortki::DefaultCpuExecutionProvider(bool enable_arena) {
    return CPUProviderFactoryCreator::Create(enable_arena)->CreateProvider();
}
