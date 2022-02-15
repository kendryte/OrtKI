#include "kernels/tensor.h"
#include "op_executor.h"

namespace ortki {
    OrtKITensor *ortki_Cast(OrtKITensor *input, DataType dataType) {
        OpExecutor cast("Cast");
        cast.AddInput("input", input);
        cast.AddAttribute("to", static_cast<int64_t>(dataType));
        return new OrtKITensor(cast.Run()[0]);
    }
}