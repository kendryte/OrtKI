#pragma once
#include "default_providers.h"
#include "tensor.h"
#include "common.h"

namespace ort_ki
{
    #define DEFINE_NODE(node) node,
    enum BinaryOp
    {
#include "binary.def"
    };
#undef DEFINE_NODE

#define DEFINE_NODE(op) ORTKI_API(OrtKITensor*) ortki_##op(OrtKITensor * a, OrtKITensor * b);
#include "binary.def"
#undef DEFINE_NODE

    ORTKI_API(OrtKITensor*) ortki_Binary(BinaryOp op, OrtKITensor* a, OrtKITensor *b);

    ORTKI_API(OrtKITensor*) ortki_Add_t(OrtKITensor * a, OrtKITensor * b);
}