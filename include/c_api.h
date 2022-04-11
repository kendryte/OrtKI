#pragma once

#include <core/framework/data_types.h>
#include <core/framework/tensor.h>
#include "common.h"
#include "tensor.h"
#include "op_executor.h"

ORTKI_API(ortki::OrtKITensor*) make_tensor(void *buffer, ortki::DataType data_type, const int* shape, int shape_size);
ORTKI_API(ortki::OrtKITensor*) make_tensor_empty(ortki::DataType data_type, const int* shape, int shape_size);

ORTKI_API(void) tensor_dispose(ortki::OrtKITensor*);
ORTKI_API(ortki::DataType) tensor_data_type(ortki::OrtKITensor *tensor);
ORTKI_API(int) tensor_rank(ortki::OrtKITensor *tensor);
ORTKI_API(void) tensor_shape(ortki::OrtKITensor *tensor, int *output);
ORTKI_API(void*) tensor_buffer(ortki::OrtKITensor *tensor);
ORTKI_API(ortki::OrtKITensor *) tensor_to_type(ortki::OrtKITensor *tensor, ortki::DataType dataType);
ORTKI_API(void) tensor_reshape(ortki::OrtKITensor *tensor, int *shape, int size);

ORTKI_API(int) tensor_seq_size(ortki::OrtKITensorSeq *);
ORTKI_API(ortki::OrtKITensor *) tensor_seq_get_value(ortki::OrtKITensorSeq *, int index);
ORTKI_API(void) tensor_seq_dispose(ortki::OrtKITensorSeq*);

ORTKI_API(ortki::OpExecutor*) make_op_executor(const char* name);
ORTKI_API(void) op_executor_dispose(ortki::OpExecutor* executor);

ORTKI_API(BFloat16*) make_bf16(float);
ORTKI_API(float) bf16_to_float(BFloat16*);
ORTKI_API(void) bf16_dispose(BFloat16*);
ORTKI_API(uint16_t) bf16_to_uint16(MLFloat16*);

ORTKI_API(MLFloat16*) make_fp16(float);
ORTKI_API(float) fp16_to_float(MLFloat16*);
ORTKI_API(void) fp16_dispose(MLFloat16*);
ORTKI_API(uint16_t) fp16_to_uint16(MLFloat16*);
