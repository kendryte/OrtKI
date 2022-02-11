#pragma once
#include <vector>
#include <core/framework/tensor_shape.h>

namespace ortki {
    inline std::vector<int64_t> GetShapeVector(const onnxruntime::TensorShape &shape) {
        std::vector<int64_t> result;
        const auto dims = shape.GetDims();
        result.resize(dims.size());
        result.assign(dims.cbegin(), dims.cend());
        return result;
    }
}