#include "c_api.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/ort_value.h>
#include "allocator_manager.h"
#include "operators.h"

using namespace ortki;
using namespace onnxruntime;

OrtKITensor* make_tensor(void* buffer, DataType data_type, const int64_t* shape, size_t rank)
{
    OrtValue value;
    Tensor::InitOrtValue(GetDataType(data_type), { shape,rank }, buffer, OrtMemoryInfo(), value);
    return new OrtKITensor(value);
}

OrtKITensor* make_tensor_empty(DataType data_type, const int64_t* shape, size_t rank)
{
    OrtValue value;
    auto allocator = ortki::AllocatorManager::Instance().GetAllocator(CPU);
    Tensor::InitOrtValue(GetDataType(data_type), { shape,rank }, allocator, value);
    auto tensor = value.GetMutable<Tensor>();
    auto data = tensor->MutableDataRaw();
    memset(data, 0, tensor->SizeInBytes());
    return new OrtKITensor(value);
}

void tensor_dispose(OrtKITensor* t)
{
    delete t;
}

DataType tensor_data_type(OrtKITensor* tensor)
{
    return tensor->data_type();
}

size_t tensor_rank(OrtKITensor* tensor)
{
    return tensor->tensor().Shape().NumDimensions();
}

size_t tensor_length(OrtKITensor* tensor)
{
    return tensor->tensor().Shape().Size();
}

void tensor_shape(OrtKITensor* tensor, int64_t* output)
{
    tensor->tensor().Shape().CopyDims(output, tensor->tensor().Shape().NumDimensions());
}

void* tensor_buffer(OrtKITensor* tensor, size_t* bytes)
{
    *bytes = tensor->tensor().SizeInBytes();
    return tensor->tensor().MutableDataRaw();
}

ortki::OpExecutor* make_op_executor(const char* name)
{
    return new OpExecutor(name);
}

// onnxruntime::Tensor don't support directly type cast
OrtKITensor* tensor_to_type(OrtKITensor* tensor, ortki::DataType dataType)
{
    return ortki_Cast(tensor, dataType);
}

void tensor_reshape(ortki::OrtKITensor* tensor, int64_t* shape, size_t size)
{
    tensor->tensor().Reshape({ shape,size });
}

void op_executor_dispose(ortki::OpExecutor* executor)
{
    delete executor;
}

size_t tensor_seq_size(ortki::OrtKITensorSeq* seq)
{
    return seq->size();
}

ortki::OrtKITensor* tensor_seq_get_value(ortki::OrtKITensorSeq* seq, size_t index)
{
    return new OrtKITensor(seq->at(index));
}

void tensor_seq_dispose(ortki::OrtKITensorSeq* seq)
{
    delete seq;
}

BFloat16* make_bf16(float v)
{
    return new BFloat16(v);
}

float bf16_to_float(BFloat16* v)
{
    return v->ToFloat();
}

void bf16_dispose(BFloat16* v)
{
    delete v;
}

uint16_t bf16_to_uint16(BFloat16* v)
{
    return v->val;
}

MLFloat16* make_fp16(float v)
{
    return new MLFloat16(v);
}

float fp16_to_float(MLFloat16* v)
{
    return v->ToFloat();
}

void fp16_dispose(MLFloat16* v)
{
    delete v;
}

uint16_t fp16_to_uint16(MLFloat16* v)
{
    return v->val;
}
