#include "kernels/binary.h"
#include "op_executor.h"

namespace ort_ki {
#define DEFINE_NODE(op) OrtKITensor *ortki_##op(OrtKITensor * a, OrtKITensor * b) \
    { \
        OpExecutor e(#op); \
        e.AddInput("A", a); \
        e.AddInput("B", b); \
        return new OrtKITensor(e.Run()[0]); \
    }
#include "kernels/binary.def"
#undef DEFINE_NODE

    OrtKITensor * ortki_Binary(BinaryOp op, OrtKITensor * a, OrtKITensor * b)
    {
#define DEFINE_NODE(op) case op: \
            return ortki_##op(a, b);
        switch(op)
        {
#include "kernels/binary.def"
            default:
                throw std::runtime_error("Unsupported BinaryOp");
        };
#undef DEFINE_NODE
    }

    OrtKITensor *ortki_Add_t(OrtKITensor * a, OrtKITensor * b) {
        std::cout << "add add tensor" << std::endl;
        OpExecutor add("Add");
        add.AddInput("A", a);
        add.AddInput("B", b);
        std::cout << "add run" << std::endl;
        add.Run();
        auto fetch = add.GetFetches()[0];
        auto &tensor = fetch.Get<onnxruntime::Tensor>();
        std::cout << tensor.Data<int>()[0] << std::endl;
        std::cout << tensor.Data<int>()[1] << std::endl;
        std::cout << tensor.Data<int>()[2] << std::endl;
        std::cout << add.GetOutputData()[0].data_.Type() << std::endl;
        std::cout << "run end" << std::endl;
        return new OrtKITensor(fetch);
    }
}