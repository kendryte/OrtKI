#pragma once
#include "common.h"
#include "util.h"
#include <core/framework/ort_value.h>
#include <core/framework/tensor_shape.h>

namespace ortki {
    struct OrtKITensor {
        // auto release data, used for create output without copy
        OrtKITensor(OrtValue handler) : _handler(handler), _tensor(handler.GetMutable<onnxruntime::Tensor>()), _owner(true) {}

        // don't auto release
        OrtKITensor(void *buffer, DataType data_type, const std::vector<int64_t>& shape)
        : OrtKITensor(buffer, data_type, onnxruntime::TensorShape(shape)){}

        // don't auto release, used for be called with other language
        OrtKITensor(void *buffer, DataType data_type, onnxruntime::TensorShape shape, bool owner = false) : _owner(owner)
        {
            _tensor = new onnxruntime::Tensor(GetDataType(data_type), shape, buffer, OrtMemoryInfo());
            auto tensor_type = onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>();
            if(owner)
            {
                _handler.Init(_tensor, tensor_type, [](auto&& p){ delete p; });
            }
            else
            {
                _handler.Init(_tensor, tensor_type, [](auto&&){});
            }
        }

        template<typename T>
        T* buffer() {
            return _tensor->MutableData<T>();
        }

        template<typename T>
        std::vector<T> to_vector() {
            auto *data = buffer<T>();
            std::vector<T> dataVec(data, data + length());
            return dataVec;
        }

        DataType data_type() const {
            return static_cast<DataType>(_tensor->GetElementType());
        }

        std::vector<int64_t> shape() const {
            return GetShapeVector(_tensor->Shape());
        }

        void reshape(const std::vector<int64_t> &new_shape) const {
            _tensor->Reshape(onnxruntime::TensorShape(new_shape));
        }

        size_t length() const {
            auto &&shape = _tensor->Shape().GetDims();
            return ComputeSize(shape);
        }

    // private:
        onnxruntime::Tensor *_tensor;
        OrtValue _handler;
        bool _owner;
    };

    // be used for create tensor array, OrtValue lifetime managed by OrtKITensor
    struct OrtKITensorSeq {
    public:
        OrtKITensorSeq(const std::vector<OrtValue>& values) : _values(values) {}

        OrtKITensor *get_value(int index) {
            return new OrtKITensor(_values[index]);
        }

        int size() const { return _values.size(); }
    private:
        std::vector<OrtValue> _values;
    };
}
