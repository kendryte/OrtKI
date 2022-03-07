#include "op_executor.h"
#include "util.h"
#include "c_api.h"
#include "operators.h"

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


template<typename T = int>
OrtKITensor *make_tensor(const std::vector<T> &value, const std::vector<int64_t> &shape)
{
    auto *ptr = new T[value.size()];
    for (int i = 0; i < value.size(); ++i) {
        ptr[i] = value[i];
    }
    return new OrtKITensor((void *)ptr, TypeToDataType<T>(), shape);
}

template<typename T = int>
OrtKITensor *make_tensor(const std::vector<T> &value)
{
    return make_tensor(value, std::vector<int64_t>{(int64_t)value.size()});
}

void test_cast()
{
    auto int_tensor = make_tensor();
    auto float_tensor = ortki_Cast(int_tensor, DataType::TensorProto_DataType_FLOAT);
    auto fp16 = ortki_Cast(float_tensor, DataType::TensorProto_DataType_FLOAT16);
    auto back_to_fp32 = ortki_Cast(fp16, DataType::TensorProto_DataType_FLOAT);
}

void test_slice()
{
    auto input = make_tensor({3});
    auto start = make_tensor({0});
    auto end = make_tensor({1});
    auto axes = make_tensor({0});
    auto steps = make_tensor({1});
    auto slice_tensor = ortki_Slice(input, start, end, axes, steps);
    std::cout << "end";
}

void test_hardswish()
{
    ortki_HardSwish(make_tensor(std::vector<float>{1.0f}));
}
void test_transpose()
{
    auto input = make_tensor({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    std::vector<int64_t> perm = {1, 0};
    ortki_Transpose(input, perm.data(), perm.size());
}

void test_split()
{
    auto input = make_tensor({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});
    std::vector<int64_t> splits_v = {1, 3};
    auto splits = make_tensor(splits_v);
    auto output = ortki_Split(input, splits, 0);
//    auto is_seq = output[0]->_handler.IsTensorSequence();
    std::cout << "str" << std::endl;
}

void test_resize()
{
    std::vector<float> X = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> Roi{};
    std::vector<int64_t> Sizes{};
    std::vector<float> Scales{1.0f, 1.0f, 0.6f, 0.6f};
    auto input = make_tensor(X, {1, 1, 2, 4});
    auto roi = make_tensor(Roi, {1});
    auto scales = make_tensor(Scales);
    auto sizes = make_tensor(Sizes);
    auto *scales_ptr = scales->buffer<float>();
    if(scales->length() == 4 && scales_ptr[0] == 1 && scales_ptr[1] == 1)
    {
        std::cout << "valid" << std::endl;
    }
    ortki_ResizeWithScales(input, roi, scales, "align_corners", -0.75, 0, 0, "linear", "floor");
}

void test_batchnorm()
{
    std::vector<float> X{0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
                    -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
                    0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
                    -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
                    -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f};
    std::vector<float> scale{0.589433f};
    std::vector<float> B{-0.384622f};
    std::vector<float> mean{-2.45673f};
    std::vector<float> var{1.37998f};

    ortki_BatchNormalization(make_tensor(X, {1, 1, 7, 7, 1}), make_tensor(scale), make_tensor(B), make_tensor(mean), make_tensor(var), 1e-05f, 0.1f);
}

void test_argmin()
{
    auto ret = ortki_ArgMin(make_tensor({1, 1, 3, 3}), 0, 1, 0);
    auto v = ret->buffer<long>();
    std::cout << "test";
}

void test_squeeze()
{
    auto ret = ortki_Squeeze(make_tensor({1, 1, 3, 3}), make_tensor({0L, 1L}));
    auto v = ret->buffer<long>();
    std::cout << "test";
}
// todo:these add to test, to different type buffer
int main()
{
     test_slice();
//    test_squeeze();
//    test_argmin();
    // test_batchnorm();
    // test_split();
    // test_transpose();
//    test_cast();
//    test_resize();
//    test_slice();

//    auto tensorA = make_tensor();
//    auto tensorB = make_tensor();
//    auto tensorC = ortki::ortki_Binary(ortki::Add, tensorA, tensorB);
//    // max(tensorA, tensorB);
//    auto max_tensor = ortki::ortki_Binary(ortki::Max, tensorA, tensorB);
//    std::cout << "value:" << tensorC->buffer<int>()[0] << std::endl;
//    std::cout << "value:" << tensorC->buffer<int>()[1] << std::endl;
//    std::cout << "value:" << tensorC->buffer<int>()[2] << std::endl;
//    auto tensor_cast = ortki::ortki_Cast(tensorC, DataType::TensorProto_DataType_FLOAT);
//    auto t2t = tensor_to_type(tensorC, DataType::TensorProto_DataType_FLOAT);
//    std::cout << t2t->buffer<float>()[0] << std::endl;

}