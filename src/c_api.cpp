#include "c_api.h"
//#include "kernels/binary.h"

OrtKITensor* make_tensor(uint8_t *buffer, DataType data_type, const int* shape, int shape_size, const int *stride)
{
    return new OrtKITensor(buffer, data_type, shape, shape_size, stride);
}

void tensor_dispose(OrtKITensor* t)
{
    delete t;
}

DataType tensor_data_type(OrtKITensor *tensor) { return tensor->data_type(); }

int tensor_rank(OrtKITensor *tensor) { return tensor->shape().size(); }

void tensor_shape(OrtKITensor *tensor, int *output)
{
    auto &shape = tensor->shape();
    for(int i = 0; i < shape.size(); ++i)
    {
        output[i] = shape[i];
    }
}

void tensor_stride(OrtKITensor *tensor, int *output)
{
    auto &stride = tensor->stride();
    for(int i = 0; i < stride.size(); ++i)
    {
        output[i] = stride[i];
    }
}

int add(OrtKITensor *a, OrtKITensor *b, OrtKITensor *output)
{
    *((int*)output->_buffer) = *((int*)a->_buffer) + *((int*)b->_buffer);
    return *((int*)output->_buffer);
}