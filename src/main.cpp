#include "op_executor.h"

int main()
{
    ort_ki::OpExecutor op("Cast", 13);
    const std::vector<int> input_int_values{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const TensorShape shape{2, 3, 2, 2};
    const size_t size = gsl::narrow<size_t>(shape.Size());
    using SrcType = int;
    using DstType = float;
    auto input_buffer = std::make_unique<SrcType[]>(size);
    auto input = gsl::make_span<SrcType>(input_buffer.get(), size);

    auto output_buffer = std::make_unique<DstType[]>(size);
    auto output = gsl::make_span<DstType>(output_buffer.get(), size);

    op.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
    auto dimensions = shape.GetDimsAsVector();
    op.AddInput<SrcType>("input", dimensions, input.data(), input.size());
    op.AddOutput<DstType>("output", dimensions, output.data(), output.size());
    op.Run();
}

//using namespace onnxruntime;
//template <typename SrcType,
//        typename DstType>
//void TestCastOp(gsl::span<const SrcType> input,
//                gsl::span<const DstType> output,
//                const std::vector<int64_t> &dimensions,
//                OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
//                const std::string& expected_failure_string = "") {
//    OpTester op("Cast", 13);
//    test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<DstType>());
//    test.AddInput<SrcType>("input", dimensions, input.data(), input.size());
//    test.AddOutput<DstType>("output", dimensions, output.data(), output.size());
//
//    std::unordered_set<std::string> excluded_provider_types{kCpuExecutionProvider};
//
//    test.Run(expect_result, expected_failure_string, excluded_provider_types);
//}

//int main()
//{
//    const std::vector<int64_t> shape{2, 2, 2};
//    const std::vector<std::string> int_16_string_data = {"0", "1", "2", "3", "4", "5", "-32768", "32767"};
//    const std::vector<int16_t> int_16_output = {0, 1, 2, 3, 4, 5, SHRT_MIN, SHRT_MAX};
//    TestCastOp(gsl::make_span(int_16_string_data), gsl::make_span(int_16_output), shape);
//
//    //auto a = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
////    auto &&info = make_kernel_info();
////    auto &&ctx = make_op_kernel_context();
////    onnxruntime::Gather gather(info);
////    gather.Compute(&ctx);
//}