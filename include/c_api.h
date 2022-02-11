#pragma once

#include <core/framework/data_types.h>
#include <core/framework/tensor.h>
#include "common.h"
#include "tensor.h"
#include "op_executor.h"

ORTKI_API(ort_ki::OrtKITensor*) make_tensor(void *buffer, DataType data_type, const int* shape, int shape_size);
ORTKI_API(void) tensor_dispose(ort_ki::OrtKITensor*);
ORTKI_API(DataType) tensor_data_type(ort_ki::OrtKITensor *tensor);
ORTKI_API(int) tensor_rank(ort_ki::OrtKITensor *tensor);
ORTKI_API(void) tensor_shape(ort_ki::OrtKITensor *tensor, int *output);

ORTKI_API(ort_ki::OpExecutor*) make_op_executor(const char* name);
ORTKI_API(void) op_executor_dispose(ort_ki::OpExecutor* executor);