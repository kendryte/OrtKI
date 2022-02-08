#include "op_executor.h"
#include "kernels/cast.h"
#include "kernels/binary.h"
#include "kernels/unary.h"
#include "util.h"

OrtKITensor *make_tensor()
{
    auto input_buffer = new int[3];
    input_buffer[0] = 2;
    input_buffer[1] = 3;
    input_buffer[2] = 4;
    std::vector<int> shape = {3};
    std::vector<int> stride = {1};
    return new OrtKITensor(reinterpret_cast<uint8_t*>(input_buffer),
                       DataType::TensorProto_DataType_INT32, shape.data(), 1, stride.data());
}
int main()
{
    auto tensorA = make_tensor();
    std::cout << "value:" << *tensorA->buffer<int>() << std::endl;
    auto tensorB = make_tensor();
    std::cout << "value:" << *tensorB->buffer<int>() << std::endl;
    auto tensorC = make_tensor();
    ort_ki::Add(tensorA, tensorB, tensorC);
    std::cout << "value:" << *tensorC->buffer<int>() << std::endl;
//    ort_ki::OpExecutor op("Cast", 13);
//    const std::vector<int> input_int_values{
//            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
//            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
//    const TensorShape shape{2, 3, 2, 2};
//    const size_t size = gsl::narrow<size_t>(shape.Size());
//    using SrcType = int;
//    using DstType = float;
//    auto input_buffer = std::make_unique<SrcType[]>(size);
//    auto input = gsl::make_span<SrcType>(input_buffer.get(), size);
//
//    auto output_buffer = std::make_unique<DstType[]>(size);
//    auto output = gsl::make_span<DstType>(output_buffer.get(), size);
//
//    op.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
//    auto dimensions = ort_ki::GetShapeVector(shape);
//    op.AddInput<SrcType>("input", dimensions, input.data(), input.size());
//    op.AddOutput<DstType>("output", dimensions, output.data(), output.size());
//    op.Run();
}