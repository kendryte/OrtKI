#pragma once

#include <core/framework/data_types.h>
#include <core/framework/tensor.h>
#include "common.h"
#include "tensor.h"

ORTKI_API(OrtKITensor*) make_tensor(uint8_t *buffer, DataType data_type, const int* shape, int shape_size, const int *stride);
ORTKI_API(void) tensor_dispose(OrtKITensor*);
ORTKI_API(DataType) tensor_data_type(OrtKITensor *tensor);
ORTKI_API(int) tensor_rank(OrtKITensor *tensor);
ORTKI_API(void) tensor_shape(OrtKITensor *tensor, int *output);
ORTKI_API(void) tensor_stride(OrtKITensor *tensor, int *output);

ORTKI_API(int) add(OrtKITensor *a, OrtKITensor *b, OrtKITensor *output);