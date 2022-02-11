#pragma once

#include <core/framework/data_types.h>
#include <core/framework/tensor.h>
#include "common.h"
#include "tensor.h"
#include "op_executor.h"

ORTKI_API(ortki::OrtKITensor*) make_tensor(void *buffer, ortki::DataType data_type, const int* shape, int shape_size);
ORTKI_API(void) tensor_dispose(ortki::OrtKITensor*);
ORTKI_API(ortki::DataType) tensor_data_type(ortki::OrtKITensor *tensor);
ORTKI_API(int) tensor_rank(ortki::OrtKITensor *tensor);
ORTKI_API(void) tensor_shape(ortki::OrtKITensor *tensor, int *output);
ORTKI_API(int*) tensor_buffer(ortki::OrtKITensor *tensor);

ORTKI_API(ortki::OpExecutor*) make_op_executor(const char* name);
ORTKI_API(void) op_executor_dispose(ortki::OpExecutor* executor);