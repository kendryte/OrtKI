#include "op_executor.h"
#include "kernels/binary.h"
#include "kernels/unary.h"
#include "kernels/tensor.h"
#include "util.h"
#include "c_api.h"

using namespace ortki;
OrtKITensor *make_tensor()
{
    auto input_buffer = new int[3];
    input_buffer[0] = 2;
    input_buffer[1] = 3;
    input_buffer[2] = 4;
    std::vector<int64_t> shape = {3};
    return new OrtKITensor(reinterpret_cast<void*>(input_buffer),
                       DataType::TensorProto_DataType_INT32, shape);
}

// todo:these add to test, to different type buffer
int main()
{
    auto tensorA = make_tensor();
    auto tensorB = make_tensor();
    auto tensorC = ortki::ortki_Binary(ortki::Add, tensorA, tensorB);
    std::cout << "value:" << tensorC->buffer<int>()[0] << std::endl;
    std::cout << "value:" << tensorC->buffer<int>()[1] << std::endl;
    std::cout << "value:" << tensorC->buffer<int>()[2] << std::endl;
    auto tensor_cast = ortki::ortki_Cast(tensorC, DataType::TensorProto_DataType_FLOAT);
    auto t2t = tensor_to_type(tensorC, DataType::TensorProto_DataType_FLOAT);
    std::cout << t2t->buffer<float>()[0] << std::endl;
}