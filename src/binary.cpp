#include "kernels/binary.h"
#include "op_executor.h"

namespace ort_ki {
    void binary(OrtKITensor * a, OrtKITensor * b, OrtKITensor * c)
    {

    }
    void Add(OrtKITensor * a, OrtKITensor * b, OrtKITensor * c) {
        std::cout << "add add tensor" << std::endl;
        OpExecutor add("Add");
        add.AddInput("a", a);
        add.AddInput("b", b);
        // add.AddOutput("output", c);
        std::cout << "add run" << std::endl;
        add.Run();
        std::cout << onnxruntime::DataTypeImpl::GetType<int>() << std::endl;
        std::cout << add.GetOutputData()[0].data_.Type() << std::endl;
        std::cout << "run end" << std::endl;
    }
}