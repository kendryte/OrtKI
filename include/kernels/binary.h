#pragma once
#include "default_providers.h"
#include "tensor.h"
#include "common.h"

namespace ort_ki
{
    #define DEFINE_NODE(node) node,
    enum class BinaryOp
    {
#include "binary.def"
    };
#undef DEFINE_NODE

    ORTKI_API(void) Binary(OrtKITensor* a, OrtKITensor *b, OrtKITensor *c);

    ORTKI_API(void) Add(OrtKITensor* a, OrtKITensor *b, OrtKITensor *c);
}