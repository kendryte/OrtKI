#pragma once
#include "common.h"
#include "util.h"
#include <core/framework/ort_value.h>
#include <core/framework/tensor_shape.h>

namespace ortki {
    struct OrtKITensor {
        // auto release data, used for create output without copy
        OrtKITensor(OrtValue handler) : _handler(handler), _tensor(handler.GetMutable<onnxruntime::Tensor>()) {}

        // don't auto release
        OrtKITensor(void *buffer, DataType data_type, const std::vector<int64_t>& shape)
        : OrtKITensor(buffer, data_type, onnxruntime::TensorShape(shape)){}

        // don't auto release, used for be called with other language
        OrtKITensor(void *buffer, DataType data_type, onnxruntime::TensorShape shape)
        {
            _tensor = new onnxruntime::Tensor(GetDataType(data_type), shape, buffer, OrtMemoryInfo());
            _handler.Init(_tensor, onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), [](auto&&){});
        }

        template<typename T>
        T* buffer() {
            return _tensor->MutableData<T>();
        }

        DataType data_type() const {
            return static_cast<DataType>(_tensor->GetElementType());
        }

        std::vector<int64_t> shape() const {
            return GetShapeVector(_tensor->Shape());
        }

        size_t length() const {
            auto &&shape = _tensor->Shape().GetDims();
            return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        }

        onnxruntime::Tensor *_tensor;
        OrtValue _handler;
    };
}
