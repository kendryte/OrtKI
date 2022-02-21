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
    auto uint8_tensor = ortki_Cast(float_tensor, DataType::TensorProto_DataType_UINT8);
}

void test_slice()
{
    auto input = make_tensor({1, 2, 3, 4, 5, 6});
    auto start = make_tensor({1});
    auto end = make_tensor({5});
    auto axes = make_tensor({0});
    auto steps = make_tensor({2});
    auto slice_tensor = ortki_Slice(input, start, end, axes, steps);

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

// todo:these add to test, to different type buffer
int main()
{
    // test_slice();
    test_split();

//    test_slice();
//    test_cast();
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