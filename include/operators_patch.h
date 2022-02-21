#include "op_executor.h"
#include "tensor.h"

// ops which be Not suitable for auto-generated
inline ortki::OrtKITensorSeq * ortki_Split(ortki::OrtKITensor * input, ortki::OrtKITensor * split, long axis)
{
    ortki::OpExecutor Split("Split");
    Split.AddInput("input", input);
    Split.AddInput("split", split);
    Split.AddAttribute("axis", axis);
    Split.SetOutputSize(split->shape()[0]);
    return new ortki::OrtKITensorSeq(Split.Run());
}