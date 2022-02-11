#include "kernels/unary.h"
#include "op_executor.h"

namespace ortki {
#define DEFINE_NODE(op) OrtKITensor *ortki_##op(OrtKITensor * input) \
    { \
        OpExecutor e(#op); \
        e.AddInput("input", input); \
        return new OrtKITensor(e.Run()[0]); \
    }
#include "kernels/unary.def"
#undef DEFINE_NODE

OrtKITensor * ortki_Unary(UnaryOp op, OrtKITensor * input)
{
#define DEFINE_NODE(op) case op: \
            return ortki_##op(input);
    switch(op)
    {
#include "kernels/unary.def"
        default:
            throw std::runtime_error("Unsupported UnaryOp");
    };
#undef DEFINE_NODE
}

}