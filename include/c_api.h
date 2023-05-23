#pragma once

#include "common.h"
#include <string>

namespace ortki
{
    class OrtKITensor;
    class OrtKITensorSeq;
    class OpExecutor;

    enum DataType : int {
        DataType_UNDEFINED = 0,
        DataType_FLOAT = 1,
        DataType_UINT8 = 2,
        DataType_INT8 = 3,
        DataType_UINT16 = 4,
        DataType_INT16 = 5,
        DataType_INT32 = 6,
        DataType_INT64 = 7,
        DataType_STRING = 8,
        DataType_BOOL = 9,
        DataType_FLOAT16 = 10,
        DataType_DOUBLE = 11,
        DataType_UINT32 = 12,
        DataType_UINT64 = 13,
        DataType_COMPLEX64 = 14,
        DataType_COMPLEX128 = 15,
        DataType_BFLOAT16 = 16
    };

    template<typename T>
    constexpr DataType TypeToDataType();

    template<>
    constexpr DataType TypeToDataType<float>() {
        return DataType_FLOAT;
    }

    template<>
    constexpr DataType TypeToDataType<double>() {
        return DataType_DOUBLE;
    }

    template<>
    constexpr DataType TypeToDataType<int32_t>() {
        return DataType_INT32;
    }

    template<>
    constexpr DataType TypeToDataType<int64_t>() {
        return DataType_INT64;
    }

    template<>
    constexpr DataType TypeToDataType<bool>() {
        return DataType_BOOL;
    }

    template<>
    constexpr DataType TypeToDataType<int8_t>() {
        return DataType_INT8;
    }

    template<>
    constexpr DataType TypeToDataType<int16_t>() {
        return DataType_INT16;
    }

    template<>
    constexpr DataType TypeToDataType<uint8_t>() {
        return DataType_UINT8;
    }

    template<>
    constexpr DataType TypeToDataType<uint16_t>() {
        return DataType_UINT16;
    }

    template<>
    constexpr DataType TypeToDataType<uint32_t>() {
        return DataType_UINT32;
    }

    template<>
    constexpr DataType TypeToDataType<uint64_t>() {
        return DataType_UINT64;
    }

    template<>
    constexpr DataType TypeToDataType<std::string>() {
        return DataType_STRING;
    }
}

ORTKI_API(ortki::OrtKITensor*) make_tensor(void* buffer, ortki::DataType data_type, const int64_t* shape, size_t shape_size);
ORTKI_API(ortki::OrtKITensor*) make_tensor_empty(ortki::DataType data_type, const int64_t* shape, size_t shape_size);

ORTKI_API(void) tensor_dispose(ortki::OrtKITensor* tensor);
ORTKI_API(ortki::DataType) tensor_data_type(ortki::OrtKITensor* tensor);
ORTKI_API(size_t) tensor_rank(ortki::OrtKITensor* tensor);
ORTKI_API(size_t) tensor_length(ortki::OrtKITensor* tensor);
ORTKI_API(void) tensor_shape(ortki::OrtKITensor* tensor, int64_t* output);
ORTKI_API(void*) tensor_buffer(ortki::OrtKITensor* tensor, size_t* bytes);
ORTKI_API(ortki::OrtKITensor*) tensor_to_type(ortki::OrtKITensor* tensor, ortki::DataType dataType);
ORTKI_API(void) tensor_reshape(ortki::OrtKITensor* tensor, int64_t* shape, size_t size);

ORTKI_API(size_t) tensor_seq_size(ortki::OrtKITensorSeq*);
ORTKI_API(ortki::OrtKITensor*) tensor_seq_get_value(ortki::OrtKITensorSeq*, size_t index);
ORTKI_API(void) tensor_seq_dispose(ortki::OrtKITensorSeq*);

ORTKI_API(ortki::OpExecutor*) make_op_executor(const char* name);
ORTKI_API(void) op_executor_dispose(ortki::OpExecutor* executor);
