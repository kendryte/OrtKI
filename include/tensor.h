#pragma once
using DataType = ONNX_NAMESPACE::TensorProto_DataType;

struct OrtKITensor
{
    OrtKITensor(uint8_t *buffer, DataType data_type, const int *shape, int rank, const int *stride)
    : _buffer(buffer), _data_type(data_type), _shape(shape, shape + rank), _stride(stride, stride + rank) {}

    template<typename T>
    T* buffer() const {
        return reinterpret_cast<T*>(_buffer);
    }

    DataType data_type() const {
        return _data_type;
    }

    const std::vector<int64_t> &shape() const {
        return _shape;
    }

    const std::vector<int64_t> &stride() const {
        return _stride;
    }

    size_t length() const {
        return std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());
    }

    uint8_t *_buffer;
    DataType _data_type;
    std::vector<int64_t> _shape;
    std::vector<int64_t> _stride;
};
