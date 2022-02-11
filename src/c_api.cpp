#include "c_api.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/ort_value.h>
//#include "kernels/binary.h"

using namespace ortki;
OrtKITensor* make_tensor(void *buffer, DataType data_type, const int* shape, int rank)
{
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

int* tensor_buffer(ortki::OrtKITensor *tensor)
{
    return tensor->buffer<int>();
}

ortki::OpExecutor *make_op_executor(const char* name)
{
    return new OpExecutor(name);
}

void op_executor_dispose(ortki::OpExecutor* executor)
{
    delete executor;
}