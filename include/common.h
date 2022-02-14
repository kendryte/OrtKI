#pragma once

namespace ortki
{
#define DEFAULT_OPSET 15

#ifdef _WIN32
    #include <intrin.h>
#define ORTKI_API(ret) extern "C" __declspec(dllexport) ret
#else
#define ORTKI_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif

    using DataType = ONNX_NAMESPACE::TensorProto_DataType;

    inline onnxruntime::MLDataType GetDataType(DataType data_type)
    {
#define GET_TYPE(tensor_type, T) \
        case onnx::TensorProto_DataType_##tensor_type: \
            return onnxruntime::DataTypeImpl::GetType<T>();

#define GET_UNIMPL_TYPE(tensor_type) \
        case onnx::TensorProto_DataType_##tensor_type: \
            throw std::runtime_error("Unimplemented input type in OpExecutor::AddInput"); \

        onnxruntime::DataTypeImpl::GetType<int>();
        switch(data_type)
        {
            GET_UNIMPL_TYPE(UNDEFINED);
            GET_TYPE(FLOAT, float);
            GET_TYPE(UINT8, uint8_t);
            GET_TYPE(INT8, int8_t);
            GET_TYPE(UINT16, uint16_t);
            GET_TYPE(INT16, int16_t);
            GET_TYPE(INT32, int32_t);
            GET_TYPE(INT64, int64_t);
            GET_UNIMPL_TYPE(STRING);
            GET_TYPE(BOOL, bool);
            GET_UNIMPL_TYPE(FLOAT16);
            GET_TYPE(DOUBLE, double);
            GET_TYPE(UINT32, uint32_t);
            GET_TYPE(UINT64, uint64_t);
            GET_UNIMPL_TYPE(COMPLEX64);
            GET_UNIMPL_TYPE(COMPLEX128);
            GET_UNIMPL_TYPE(BFLOAT16);
        }
        throw std::runtime_error("Unsupported DataType");
#undef GET_TYPE
    }
}
