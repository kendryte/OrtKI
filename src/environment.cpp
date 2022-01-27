#include <core/session/onnxruntime_cxx_api.h>
#include "environment.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/ort_env.h"
#include "core/session/environment.h"

using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;

namespace ort_ki
{
    static std::unique_ptr<::onnxruntime::logging::LoggingManager> s_default_logging_manager;

    const ::onnxruntime::Environment& GetEnvironment() {
        return ((OrtEnv*)*ort_env.get())->GetEnvironment();
    }

    ::onnxruntime::logging::LoggingManager& DefaultLoggingManager() {
        return *((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager();
    }
}