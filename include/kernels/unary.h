#pragma once
#include "default_providers.h"
#include "tensor.h"
#include "common.h"

namespace ortki
{
#define DEFINE_NODE(node) node,
    enum UnaryOp
    {
#include "unary.def"
    };
#undef DEFINE_NODE

#define DEFINE_NODE(op) ORTKI_API(OrtKITensor*) ortki_##op(OrtKITensor *input);
#include "unary.def"
#undef DEFINE_NODE

ORTKI_API(OrtKITensor*) ortki_Unary(UnaryOp op, OrtKITensor* input);

}