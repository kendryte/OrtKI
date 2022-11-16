#include "util.h"
#include "tensor.h"

ONNX_NAMESPACE::TensorProto ortki::ToTensor(OrtKITensor* tensor) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    // 1. set dimension
    auto& shape = tensor->tensor().Shape();
    for (auto& dim : shape.GetDims()) {
        tensor_proto.add_dims(dim);
    }
    // 2. set type
    auto elemtype = tensor->tensor().GetElementType();
    tensor_proto.set_data_type(elemtype);
    // 3. data
    if (elemtype == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
        const std::string* string_data = tensor->tensor().Data<std::string>();
        for (auto i = 0; i < shape.Size(); i++) {
            tensor_proto.add_string_data(string_data[i]);
        }
    }
    else {
        auto buffer_size = tensor->tensor().DataType()->Size() * shape.Size();
        tensor_proto.set_raw_data(tensor->tensor().DataRaw(), buffer_size);
    }

    return tensor_proto;
}
