#pragma once
#include "default_providers.h"

namespace ort_ki
{
#define DEFINE_NODE(node) node,
    enum class UnaryOp
    {
#include "unary.def"
    };
#undef DEFINE_NODE
}