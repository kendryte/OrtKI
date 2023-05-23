#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <memory>
#include "environment.h"
#include "core/common/logging/logging.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"
#include "core/util/thread_utils.h"

using namespace ::onnxruntime::logging;

OrtEnv *make_ort_env()
{
    OrtThreadingOptions tpo;
    OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, ORT_LOGGING_LEVEL_WARNING, "Default"};
    onnxruntime::Status status;
    return OrtEnv::GetInstance(lm_info, status, &tpo);
}

OrtEnv *ort_env = make_ort_env();
namespace ortki
{
    static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

    const ::onnxruntime::Environment& GetEnvironment() {
        return ort_env->GetEnvironment();
    }

    ::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
        return *ort_env->GetLoggingManager();
    }
}