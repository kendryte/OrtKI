#include "c_api.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/ort_value.h>
#include "kernels/tensor.h"

using namespace ortki;
OrtKITensor* make_tensor(void *buffer, DataType data_type, const int* shape, int rank)
{
    std::cout << "make tensor" << std::endl;
    std::vector<int64_t> shape_vec(shape, shape + rank);
    return new OrtKITensor(buffer, data_type, shape_vec);
}

void tensor_dispose(OrtKITensor* t)
{
    delete t;
}

DataType tensor_data_type(OrtKITensor *tensor) { return tensor->data_type(); }

int tensor_rank(OrtKITensor *tensor) { return tensor->shape().size(); }

void tensor_shape(OrtKITensor *tensor, int *output)
{
    auto &&shape = tensor->shape();
    for(int i = 0; i < shape.size(); ++i)
    {
        output[i] = shape[i];
    }
}

void* tensor_buffer(ortki::OrtKITensor *tensor)
{
#define GET_BUFFER(tensor_type, T) \
    case onnx::TensorProto_DataType_##tensor_type: \
        return reinterpret_cast<void*>(tensor->buffer<T>());                \

#define GET_UNIMPL_BUFFER(tensor_type) \
    case onnx::TensorProto_DataType_##tensor_type: \
        throw std::runtime_error("Unimplemented input type in tensor_buffer");
    DATATYPE_TO_T(tensor->data_type(), GET_BUFFER, GET_UNIMPL_BUFFER);
}

ortki::OpExecutor *make_op_executor(const char* name)
{
    return new OpExecutor(name);
}

// onnxruntime::Tensor don't support directly type cast
ortki::OrtKITensor *tensor_to_type(ortki::OrtKITensor *tensor, ortki::DataType dataType)
{
    return ortki::ortki_Cast(tensor, dataType);
}

void op_executor_dispose(ortki::OpExecutor* executor)
{
    delete executor;
}