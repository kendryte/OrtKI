#include <core/session/onnxruntime_cxx_api.h>

#include <memory>
#include "environment.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"
#include "core/util/thread_utils.h"

using namespace ::onnxruntime::logging;
// std::unique_ptr<Ort::Env> ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
std::unique_ptr<Ort::Env> make_ort_env()
{
    OrtThreadingOptions tpo;
    return std::make_unique<Ort::Env>(&tpo, ORT_LOGGING_LEVEL_WARNING, "Default");
}

std::unique_ptr<Ort::Env> ort_env = make_ort_env();

namespace ortki
{
    static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

    const ::onnxruntime::Environment& GetEnvironment() {
        return ((OrtEnv*)*ort_env.get())->GetEnvironment();
    }

    ::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
        return *((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager();
    }

    void reset_env()
    {
        ort_env.reset(nullptr);
    }

    void init_env()
    {
        try{
            ort_env = make_ort_env();
        } catch (const std::exception& e) {

        }
    }
}