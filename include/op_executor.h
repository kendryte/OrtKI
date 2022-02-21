#pragma once
#include <vector>
#include <functional>
#include <core/graph/graph.h>
#include <core/graph/model.h>
#include <core/framework/customregistry.h>
#include <core/framework/run_options.h>
#include "allocator_manager.h"
#include "common.h"
#include "tensor.h"
#include "environment.h"

using namespace onnxruntime;

namespace ortki {
    template<typename T>
    struct SeqTensors {
        void AddTensor(const std::vector<int64_t> &shape0, const std::vector<T> &data0) {
            tensors.push_back(Tensor<T>{shape0, data0});
        }

        template<typename U>
        struct Tensor {
            std::vector<int64_t> shape;
            std::vector<U> data;
        };
        std::vector<Tensor<T>> tensors;
    };

// Function templates to translate C++ types into ONNX_NAMESPACE::TensorProto_DataTypes
    template<typename T>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType();

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<float>() {
        return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<double>() {
        return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int32_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int64_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<bool>() {
        return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int8_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<int16_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint8_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint16_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint32_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<uint64_t>() {
        return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<std::string>() {
        return ONNX_NAMESPACE::TensorProto_DataType_STRING;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<MLFloat16>() {
        return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    }

    template<>
    constexpr ONNX_NAMESPACE::TensorProto_DataType TypeToDataType<BFloat16>() {
        return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    }

    template<typename T>
    struct TTypeProto {
        TTypeProto(const std::vector<int64_t> *shape = nullptr) {
            proto.mutable_tensor_type()->set_elem_type(TypeToDataType<T>());

            if (shape) {
                auto mutable_shape = proto.mutable_tensor_type()->mutable_shape();
                for (auto i: *shape) {
                    auto *mutable_dim = mutable_shape->add_dim();
                    if (i != -1)
                        mutable_dim->set_dim_value(i);
                    else
                        mutable_dim->set_dim_param("symbolic");
                }
            }
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

// Variable template for ONNX_NAMESPACE::TensorProto_DataTypes, s_type_proto<float>, etc..
    template<typename T>
    struct TTensorType {
        static const TTypeProto<T> s_type_proto;
    };

    template<typename T>
    const TTypeProto<T> TTensorType<T>::s_type_proto;

#if !defined(DISABLE_SPARSE_TENSORS)

    struct TSparseTensorProto {
        explicit TSparseTensorProto(int32_t dtype, const std::vector<int64_t> *shape = nullptr) {
            proto.mutable_sparse_tensor_type()->set_elem_type(dtype);
            if (shape) {
                auto m_shape = proto.mutable_sparse_tensor_type()->mutable_shape();
                for_each(shape->cbegin(), shape->cend(), [m_shape](int64_t v) {
                    auto *m_dim = m_shape->add_dim();
                    if (v != -1)
                        m_dim->set_dim_value(v);
                    else
                        m_dim->set_dim_param("symbolic");
                });
            }
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

#endif

// TypeProto for map<TKey, TVal>
    template<typename TKey, typename TVal>
    struct MTypeProto {
        MTypeProto() {
            proto.mutable_map_type()->set_key_type(TypeToDataType<TKey>());
            proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->set_elem_type(
                    TypeToDataType<TVal>());
            proto.mutable_map_type()->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename TKey, typename TVal>
    struct MMapType {
        static const MTypeProto<TKey, TVal> s_map_type_proto;
    };

    template<typename TKey, typename TVal>
    const MTypeProto<TKey, TVal> MMapType<TKey, TVal>::s_map_type_proto;

// TypeProto for vector<map<TKey, TVal>>
    template<typename TKey, typename TVal>
    struct VectorOfMapTypeProto {
        VectorOfMapTypeProto() {
            auto *map_type = proto.mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
            map_type->set_key_type(TypeToDataType<TKey>());
            map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TypeToDataType<TVal>());
            map_type->mutable_value_type()->mutable_tensor_type()->mutable_shape()->clear_dim();
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename TKey, typename TVal>
    struct VectorOfMapType {
        static const VectorOfMapTypeProto<TKey, TVal> s_vec_map_type_proto;
    };

    template<typename TKey, typename TVal>
    const VectorOfMapTypeProto<TKey, TVal> VectorOfMapType<TKey, TVal>::s_vec_map_type_proto;

    template<typename ElemType>
    struct SequenceTensorTypeProto {
        SequenceTensorTypeProto() {
            MLDataType dt = DataTypeImpl::GetTensorType<ElemType>();
            const auto *elem_proto = dt->GetTypeProto();
            proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(*elem_proto);
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

    template<typename ElemType>
    struct SequenceTensorType {
        static const SequenceTensorTypeProto<ElemType> s_sequence_tensor_type_proto;
    };

    template<typename ElemType>
    const SequenceTensorTypeProto<ElemType> SequenceTensorType<ElemType>::s_sequence_tensor_type_proto;

#if !defined(DISABLE_OPTIONAL_TYPE)

    template<typename ElemType>
    struct OptionalTypeProto {
        OptionalTypeProto(const ONNX_NAMESPACE::TypeProto &type_proto) {
            proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(type_proto);
        }

        ONNX_NAMESPACE::TypeProto proto;
    };

#endif

    class OpExecutor {
    public:
        // Default to the first opset that ORT was available (7).
        // When operators are updated they need to explicitly add tests for the new opset version.
        // This is due to the kernel matching logic. See KernelRegistry::VerifyKernelDef.
        // Additionally, -1 is supported and defaults to the latest known opset.
        //
        // Defaulting to the latest opset version would result in existing operator implementations for non-CPU EPs to
        // lose their test coverage until an implementation for the new version is added.
        //   e.g. there are CPU and GPU implementations for version 1 of an op. both are tested by a single OpTester test.
        //        opset changes from 1 to 2 and CPU implementation gets added. If 'opset_version' is 2 the kernel matching
        //        will find and run the CPU v2 implementation, but will not match the GPU v1 implementation.
        //        OpTester will say it was successful as at least one EP ran, and the GPU implementation of v1 no longer has
        //        test coverage.
        explicit OpExecutor(const char *op, int opset_version = DEFAULT_OPSET, const char *domain = onnxruntime::kOnnxDomain,
                            bool verify_output = true)
                : op_(op), domain_(domain), opset_version_(opset_version), verify_output_(verify_output) {
            if (opset_version_ < 0) {
                static int latest_onnx_version =
                        ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(
                                ONNX_NAMESPACE::ONNX_DOMAIN).second;
                opset_version_ = latest_onnx_version;
            }
            init_env();
        }

        ~OpExecutor()
        {
            reset_env();
        }
        // Set whether the NodeArg created by AddInput/AddOutput should include shape information
        // for Tensor types. If not added, shape inferencing should resolve. If added, shape inferencing
        // should validate. Default is to not add.
        // Additionally a symbolic dimension will be added if symbolic_dim matches a dimension in the input.
        OpExecutor &AddShapeToTensorData(bool add_shape = true, int symbolic_dim = -1) {
            add_shape_to_tensor_data_ = add_shape;
            add_symbolic_dim_to_tensor_data_ = symbolic_dim;
            return *this;
        }

        void SetOutputSize(size_t size)
        {
            output_size_ = size;
        }

        // We have an initializer_list and vector version of the Add functions because std::vector is specialized for
        // bool and we can't get the raw data out. So those cases must use an initializer_list
        template<typename T>
        void AddInput(const char *name, const std::vector<int64_t> &dims, const std::initializer_list<T> &values,
                      bool is_initializer = false, const std::vector<std::string> *dim_params = nullptr) {
            AddData(input_data_, name, dims, values.begin(), values.size(), is_initializer, false, dim_params);
        }

        template<typename T>
        void AddInput(const char *name, const std::vector<int64_t> &dims, const std::vector<T> &values,
                      bool is_initializer = false, const std::vector<std::string> *dim_params = nullptr) {
            AddData(input_data_, name, dims, values.data(), values.size(), is_initializer, false, dim_params);
        }

        template<typename T>
        void AddInput(const char *name, const std::vector<int64_t> &dims, const T *p_values,
                      const size_t size, bool is_initializer = false,
                      const std::vector<std::string> *dim_params = nullptr) {
            AddData(input_data_, name, dims, p_values, size, is_initializer, false, dim_params);
        }

        void AddInput(const std::string name, OrtKITensor *tensor)
        {
            AddInput(name.c_str(), tensor);
        }

        void AddInput(const char *name, OrtKITensor *tensor)
        {
#define ADD_INPUT(tensor_type, t) \
        case onnx::TensorProto_DataType_##tensor_type: \
            AddInput<t>(name, tensor->shape(), tensor->buffer<t>(), tensor->length()); \
            return;

#define ADD_UNIMPL_INPUT(tensor_type) \
        case onnx::TensorProto_DataType_##tensor_type: \
            throw std::runtime_error("Unimplemented input type in OpExecutor::AddInput"); \
            return;

            switch(tensor->data_type())
            {
                ADD_UNIMPL_INPUT(UNDEFINED);
                ADD_INPUT(FLOAT, float);
                ADD_INPUT(UINT8, uint8_t);
                ADD_INPUT(INT8, int8_t);
                ADD_INPUT(UINT16, uint16_t);
                ADD_INPUT(INT16, int16_t);
                ADD_INPUT(INT32, int32_t);
                ADD_INPUT(INT64, int64_t);
                ADD_UNIMPL_INPUT(STRING);
                ADD_INPUT(BOOL, bool);
                ADD_UNIMPL_INPUT(FLOAT16);
                ADD_INPUT(DOUBLE, double);
                ADD_INPUT(UINT32, uint32_t);
                ADD_INPUT(UINT64, uint64_t);
                ADD_UNIMPL_INPUT(COMPLEX64);
                ADD_UNIMPL_INPUT(COMPLEX128);
                ADD_UNIMPL_INPUT(BFLOAT16);
            }
            throw std::runtime_error("Unsupported type in OpExecutor::AddInput");
        }

#if !defined(DISABLE_SPARSE_TENSORS)

        // Useful to add boolean data
        template<typename T>
        void AddSparseCooInput(const char *name, const std::vector<int64_t> &dims,
                               const std::initializer_list<T> &values, const std::vector<int64_t> &indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(input_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(indices), dim_params);
        }

        template<typename T>
        void AddSparseCooInput(const char *name, const std::vector<int64_t> &dims,
                               const std::vector<T> &values, const std::vector<int64_t> &indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(input_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(indices), dim_params);
        }

        template<typename T>
        void AddSparseCooInput(const char *name, const std::vector<int64_t> &dims,
                               gsl::span<const T> values_span,
                               const std::vector<int64_t> &indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(input_data_, ml_type, name, dims,
                                   values_span.as_bytes(),
                                   gsl::make_span(indices), dim_params);
        }

        void AddSparseCooInput(const char *name, const std::vector<int64_t> &dims,
                               const std::vector<std::string> &values,
                               const std::vector<int64_t> &indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            AddSparseCooTensorStrings(input_data_, name, dims,
                                      gsl::make_span(values),
                                      gsl::make_span(indices),
                                      dim_params);
        }

        // Useful to add boolean data
        template<typename T>
        void AddSparseCsrInput(const char *name, const std::vector<int64_t> &dims,
                               const std::initializer_list<T> &values,
                               const std::vector<int64_t> &inner_indices,
                               const std::vector<int64_t> &outer_indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(inner_indices),
                                   gsl::make_span(outer_indices), dim_params);
        }

        template<typename T>
        void AddSparseCsrInput(const char *name, const std::vector<int64_t> &dims,
                               const std::vector<T> &values,
                               const std::vector<int64_t> &inner_indices,
                               const std::vector<int64_t> &outer_indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(inner_indices),
                                   gsl::make_span(outer_indices), dim_params);
        }

        template<typename T>
        void AddSparseCsrInput(const char *name, const std::vector<int64_t> &dims,
                               gsl::span<const T> values_span,
                               const std::vector<int64_t> &inner_indices,
                               const std::vector<int64_t> &outer_indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                                   values_span.as_bytes(),
                                   gsl::make_span(inner_indices),
                                   gsl::make_span(outer_indices), dim_params);
        }

        void AddSparseCsrInput(const char *name, const std::vector<int64_t> &dims,
                               const std::vector<std::string> &values,
                               const std::vector<int64_t> &inner_indices,
                               const std::vector<int64_t> &outer_indices,
                               const std::vector<std::string> *dim_params = nullptr) {
            AddSparseCsrTensorStrings(input_data_, name, dims,
                                      gsl::make_span(values),
                                      gsl::make_span(inner_indices),
                                      gsl::make_span(outer_indices),
                                      dim_params);
        }

#endif

        // Add other registered types, possibly experimental
        template<typename T>
        void AddInput(const char *name, const T &val) {
            auto mltype = DataTypeImpl::GetType<T>();
            ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
            auto ptr = std::make_unique<T>(val);
            OrtValue value;
            value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
            ptr.release();
            input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value)));
        }

        template<typename T>
        void AddInput(const char *name, T &&val) {
            auto mltype = DataTypeImpl::GetType<T>();
            ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
            auto ptr = std::make_unique<T>(std::move(val));
            OrtValue value;
            value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
            ptr.release();
            input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value)));
        }

        template<typename T>
        void AddSeqInput(const char *name, const SeqTensors<T> &seq_tensors) {
            AddSeqData<T>(input_data_, name, &seq_tensors);
        }

        template<typename T>
        void AddSeqOutput(const char *name, const SeqTensors<T> &seq_tensors) {
            AddSeqData<T>(output_data_, name, &seq_tensors);
        }

#if !defined(DISABLE_OPTIONAL_TYPE)

        template<typename T>
        void AddOptionalTypeTensorInput(const char *name, const std::vector<int64_t> &dims,
                                        const std::initializer_list<T> *values = nullptr,
                                        bool is_initializer = false,
                                        const std::vector<std::string> *dim_params = nullptr) {
            AddData(input_data_, name, dims, values ? values->begin() : nullptr,
                    values ? values->size() : 0, is_initializer, false, dim_params, 0.0f, 0.0f, true);
        }

        template<typename T>
        void AddOptionalTypeTensorOutput(const char *name, const std::vector<int64_t> &dims,
                                         const std::initializer_list<T> *expected_values = nullptr,
                                         bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
            AddData(output_data_, name, dims, expected_values ? expected_values->begin() : nullptr,
                    expected_values ? expected_values->size() : 0, false,
                    sort_output, nullptr /* dim_params */, rel_error, abs_error, true);
        }

        template<typename T>
        void AddOptionalTypeSeqInput(const char *name,
                                     const SeqTensors<T> *seq_tensors) {
            AddSeqData<T>(input_data_, name, seq_tensors, true);
        }

        template<typename T>
        void AddOptionalTypeSeqOutput(const char *name,
                                      const SeqTensors<T> *seq_tensors) {
            AddSeqData<T>(output_data_, name, seq_tensors, true);
        }

#endif

        template<typename TKey, typename TVal>
        void AddInput(const char *name, const std::map<TKey, TVal> &val) {
            std::unique_ptr<std::map<TKey, TVal>> ptr = std::make_unique<std::map<TKey, TVal>>(val);
            OrtValue value;
            value.Init(ptr.release(), DataTypeImpl::GetType<std::map<TKey, TVal>>(),
                       DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
            input_data_.push_back(Data(NodeArg(name, &MMapType<TKey, TVal>::s_map_type_proto.proto), std::move(value)));
        }

        /*
        * Use this API to add an input *edge* to the node/op being tested that won't
        * have any data passed into.
        * Such an edge will have the qualifier OpSchema::Optional in the schema.
        * This is exposed to ensure the op kernel implementations can be tested to handle
        * presence/absence of such optional input edges.
        */
        template<typename T>
        void AddOptionalInputEdge() {
            std::string name;  // empty == input doesn't exist
            input_data_.push_back(
                    Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue()));
        }

        template<typename T>
        void
        AddOutput(const char *name, const std::vector<int64_t> &dims, const std::initializer_list<T> &expected_values,
                  bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
            AddData(output_data_, name, dims, expected_values.begin(), expected_values.size(), false,
                    sort_output, nullptr /* dim_params */, rel_error, abs_error);
        }

        // This function doesn't work for vector<bool> because const vector<bool> cannot invoke its data().
        template<typename T>
        void AddOutput(const char *name, const std::vector<int64_t> &dims, const std::vector<T> &expected_values,
                       bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
            AddData(output_data_, name, dims, expected_values.data(), expected_values.size(), false,
                    sort_output, nullptr /* dim_params */, rel_error, abs_error);
        }

        template<typename T>
        void AddOutput(const char *name, const std::vector<int64_t> &dims, const T *p_values, const size_t size,
                       bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
            AddData(output_data_, name, dims, p_values, size, false,
                    sort_output, nullptr /* dim_params */, rel_error, abs_error);
        }

        void AddOutput(const char *name, OrtKITensor *tensor)
        {
#define ADD_OUTPUT(tensor_type, t) \
        case onnx::TensorProto_DataType_##tensor_type: \
            AddOutput<t>(name, tensor->shape(), tensor->buffer<t>(), tensor->length()); \
            return;

#define ADD_UNIMPL_OUTPUT(tensor_type) \
        case onnx::TensorProto_DataType_##tensor_type: \
            throw std::runtime_error("Unimplemented output type in OpExecutor::AddInput"); \
            return;

            switch(tensor->data_type())
            {
                ADD_UNIMPL_OUTPUT(UNDEFINED);
                ADD_OUTPUT(FLOAT, float);
                ADD_OUTPUT(UINT8, uint8_t);
                ADD_OUTPUT(INT8, int8_t);
                ADD_OUTPUT(UINT16, uint16_t);
                ADD_OUTPUT(INT16, int16_t);
                ADD_OUTPUT(INT32, int32_t);
                ADD_OUTPUT(INT64, int64_t);
                ADD_UNIMPL_OUTPUT(STRING);
                ADD_OUTPUT(BOOL, bool);
                ADD_UNIMPL_OUTPUT(FLOAT16);
                ADD_OUTPUT(DOUBLE, double);
                ADD_OUTPUT(UINT32, uint32_t);
                ADD_OUTPUT(UINT64, uint64_t);
                ADD_UNIMPL_OUTPUT(COMPLEX64);
                ADD_UNIMPL_OUTPUT(COMPLEX128);
                ADD_UNIMPL_OUTPUT(BFLOAT16);
            }
            throw std::runtime_error("Unsupported type in OpExecutor::AddOutput");
        }

#if !defined(DISABLE_SPARSE_TENSORS)

        template<typename T>
        void AddSparseCooOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::initializer_list<T> &expected_values,
                                const std::vector<int64_t> &expected_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(output_data_, ml_type, name, dims,
                                   gsl::make_span(expected_values).as_bytes(),
                                   gsl::make_span(expected_indices),
                                   nullptr /*dim_params*/);
        }

        template<typename T>
        void AddSparseCooOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::vector<T> &expected_values,
                                const std::vector<int64_t> &expected_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(output_data_, ml_type, name, dims,
                                   gsl::make_span(expected_values).as_bytes(),
                                   gsl::make_span(expected_indices),
                                   nullptr /*dim_params*/);
        }

        template<typename T>
        void AddSparseCooOutput(const char *name, const std::vector<int64_t> &dims,
                                gsl::span<const T> expected_values_span,
                                const std::vector<int64_t> &expected_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCooTensorData(output_data_, ml_type, name, dims,
                                   expected_values_span.as_bytes(),
                                   gsl::make_span(expected_indices),
                                   nullptr /*dim_params*/);
        }

        void AddSparseCooOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::vector<std::string> &expected_values,
                                const std::vector<int64_t> &expected_indices) {
            AddSparseCooTensorStrings(output_data_, name, dims,
                                      gsl::make_span(expected_values),
                                      gsl::make_span(expected_indices));
        }

        template<typename T>
        void AddSparseCsrOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::initializer_list<T> &values,
                                const std::vector<int64_t> &inner_indices,
                                const std::vector<int64_t> &outer_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(inner_indices),
                                   gsl::make_span(outer_indices),
                                   nullptr /*dim_params*/);
        }

        template<typename T>
        void AddSparseCsrOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::vector<T> &values,
                                const std::vector<int64_t> &inner_indices,
                                const std::vector<int64_t> &outer_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                                   gsl::make_span(values).as_bytes(),
                                   gsl::make_span(inner_indices),
                                   gsl::make_span(outer_indices),
                                   nullptr /*dim_params*/);
        }

        template<typename T>
        void AddSparseCsrOutput(const char *name, const std::vector<int64_t> &dims,
                                gsl::span<const T> expected_values_span,
                                const std::vector<int64_t> &expected_inner_indices,
                                const std::vector<int64_t> &expected_outer_indices) {
            auto ml_type = DataTypeImpl::GetType<T>();
            AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                                   expected_values_span.as_bytes(),
                                   gsl::make_span(expected_inner_indices),
                                   gsl::make_span(expected_outer_indices),
                                   nullptr /*dim_params*/);
        }

        void AddSparseCsrOutput(const char *name, const std::vector<int64_t> &dims,
                                const std::vector<std::string> &expected_values,
                                const std::vector<int64_t> &expected_inner_indices,
                                const std::vector<int64_t> &expected_outer_indices) {
            AddSparseCsrTensorStrings(output_data_, name, dims,
                                      gsl::make_span(expected_values),
                                      gsl::make_span(expected_inner_indices),
                                      gsl::make_span(expected_outer_indices));
        }

#endif

        /*
        * Use this API to add an output *edge* to the node/op being tested that shouldn't have any
        * data produced into.
        * Such an edge will have the qualifier OpSchema::Optional in the schema.
        * This is exposed to ensure the op kernel implementations can be tested to handle
        * presence/absence of such optional output edges.
        */
        template<typename T>
        void AddOptionalOutputEdge() {
            std::string name;  // empty == output doesn't exist
            output_data_.push_back(
                    Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue()));
        }

        // Add other registered types, possibly experimental
        template<typename T>
        void AddOutput(const char *name, const T &val) {
            auto mltype = DataTypeImpl::GetType<T>();
            ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
            auto ptr = std::make_unique<T>(val);
            OrtValue value;
            value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
            ptr.release();
            output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value)));
        }

        template<typename T>
        void AddOutput(const char *name, T &&val) {
            auto mltype = DataTypeImpl::GetType<T>();
            ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
            auto ptr = std::make_unique<T>(std::move(val));
            OrtValue value;
            value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
            ptr.release();
            output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value)));
        }

        // Add non tensor output
        template<typename TKey, typename TVal>
        void AddOutput(const char *name, const std::vector<std::map<TKey, TVal>> &val) {
            auto ptr = std::make_unique<std::vector<std::map<TKey, TVal>>>(val);
            OrtValue ml_value;
            ml_value.Init(ptr.release(), DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>(),
                          DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>()->GetDeleteFunc());
            output_data_.push_back(
                    Data(NodeArg(name, &VectorOfMapType<TKey, TVal>::s_vec_map_type_proto.proto), std::move(ml_value)));
        }

        // Generate the reference outputs with the model file
        // void AddReferenceOutputs(const std::string &model_path);

        void AddCustomOpRegistry(std::shared_ptr<CustomRegistry> registry) {
            custom_schema_registries_.push_back(registry->GetOpschemaRegistry());
            custom_session_registries_.push_back(registry);
        }

        // Number of times to call InferenceSession::Run. The same feeds are used each time.
        // e.g. used to verify the generator ops behave as expected
        void SetNumRunCalls(int n) {
            ORT_ENFORCE(n > 0);
            num_run_calls_ = n;
        }

        using CustomOutputVerifierFn =
        std::function<void(const std::vector<OrtValue> & /*fetches*/, const std::string & /*provider_type*/)>;

        void SetCustomOutputVerifier(CustomOutputVerifierFn custom_output_verifier) {
            custom_output_verifier_ = custom_output_verifier;
        }

        template<typename T>
        void AddAttribute(std::string name, T value) {
            // Generate a the proper AddAttribute call for later
            add_attribute_funcs_.emplace_back(
                    [name = std::move(name), value = std::move(value)](onnxruntime::Node &node) {
                        node.AddAttribute(name, value);
                    });
        }

        enum class ExpectResult {
            kExpectSuccess,
            kExpectFailure
        };

        std::vector<OrtValue> Run(const std::unordered_set<std::string> &excluded_provider_types = {},
            const RunOptions *run_options = nullptr,
            std::vector<std::unique_ptr<IExecutionProvider>> *execution_providers = nullptr,
            ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL);

        std::vector<OrtValue> Run(SessionOptions session_options,
                 const std::unordered_set<std::string> &excluded_provider_types = {},
                 const RunOptions *run_options = nullptr,
                 std::vector<std::unique_ptr<IExecutionProvider>> *execution_providers = nullptr,
                 const Graph::ResolveOptions &resolve_options = {},
                /*out*/ size_t *number_of_pre_packed_weights_counter = nullptr,
                /*out*/ size_t *number_of_shared_pre_packed_weights_counter = nullptr);

        std::vector<OrtValue>
        GetFetches() { return fetches_; }

        std::unique_ptr<onnxruntime::Model>
        BuildGraph(const std::unordered_map<std::string, int> &extra_domain_to_version = {},
                   bool allow_released_onnx_opset_only = true);

        // storing p_model as cache
        void SetModelCache(std::shared_ptr<onnxruntime::Model> model) {
            cached_model_ = model;
        }

        std::shared_ptr<onnxruntime::Model> GetModelCache() {
            return cached_model_;
        }

        // clear input/output data, fetches will be cleared in Run()
        void ClearData() {
            input_data_.clear();
            output_data_.clear();
            initializer_index_.clear();
        }

        struct Data {
            onnxruntime::NodeArg def_;
            OrtValue data_;

            Data(onnxruntime::NodeArg &&def, OrtValue &&data,
                 bool sort_output = false)
                    : def_(std::move(def)),
                      data_(std::move(data)) {}

            Data(Data &&) = default;

            Data &operator=(Data &&) = default;
        };

        std::vector<Data> &GetInputData() {
            return input_data_;
        }

        std::vector<Data> &GetOutputData() {
            return output_data_;
        }

        void SetDeterminism(bool use_determinism) {
            use_determinism_ = use_determinism;
        }

        void EnableSharingOfPrePackedWeightsAcrossSessions() {
            add_prepacked_shared_container_to_sessions_ = true;
        }

        size_t GetNumPrePackedWeightsShared() const {
            return prepacked_weights_container_.GetNumberOfElements();
        }

        bool test_allow_released_onnx_opset_only_ = true;

    protected:
        // Set test_allow_released_onnx_opset_only_ to false or override this method and return false
        // if inheriting from OpTester to allow testing of a non-released ONNX opset operator
        virtual bool IsAllowReleasedONNXOpsetsOnlySetForThisTest() const {
            return test_allow_released_onnx_opset_only_;
        }

        virtual void AddNodes(onnxruntime::Graph &graph, std::vector<onnxruntime::NodeArg *> &graph_input_defs,
                              std::vector<onnxruntime::NodeArg *> &graph_output_defs,
                              std::vector<std::function<void(onnxruntime::Node &node)>> &add_attribute_funcs);

        void AddInitializers(onnxruntime::Graph &graph);

        void FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue> &feeds,
                                     std::vector<std::string> &output_names);

        void FillFeeds(std::unordered_map<std::string, OrtValue> &feeds);

        template<class SessionType>
        std::vector<OrtValue> ExecuteModel(Model &model,
                                           SessionType &session_object,
                                           const RunOptions *run_options,
                                           const std::unordered_map<std::string, OrtValue> &feeds,
                                           const std::vector<std::string> &output_names,
                                           const std::string &provider_type,
                                           bool allow_released_onnx_opset_only = true);

        const char *op_;
        std::vector<Data> input_data_;
        std::vector<Data> output_data_;
        std::vector<OrtValue> fetches_;

        // for gradient unit tests only
        std::shared_ptr<onnxruntime::Model> cached_model_;

#ifndef NDEBUG
        bool run_called_{};
#endif

    protected:
        template<typename T>
        void AddData(std::vector<Data> &data, const char *name, gsl::span<const int64_t> dims, const T *values,
                     int64_t values_count, bool is_initializer = false, bool sort_output = false,
                     const std::vector<std::string> *dim_params = nullptr,
                     float rel_error = 0.0f, float abs_error = 0.0f, bool is_optional_type_tensor = false) {
#if defined(DISABLE_OPTIONAL_TYPE)
            if (is_optional_type_tensor) {
          ORT_THROW("Optional type is not supported in this build");
        }
#endif

            ORT_TRY {
                TensorShape shape{dims};

                OrtValue value;

                if (!is_optional_type_tensor || (is_optional_type_tensor && values != nullptr)) {
                    // In case values is nullptr for optional type tensor, it means we are creating
                    // an optional type tensor which is None and we hence skip values count validation
                    ORT_ENFORCE(shape.Size() == values_count,
                                values_count, " input values doesn't match tensor size of ",
                                shape.Size());

                    // If it is an optional tensor type with no values (i.e.) None,
                    // we won't even pass it in to Run() as part of the feeds,
                    // so we don't even have to create a Tensor.
                    // Conversely, if it is an optional tensor type with values,
                    // we pass it in as a regular tensor.
                    auto allocator = AllocatorManager::Instance().GetAllocator(CPU);
                    Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, std::move(allocator), value);

                    // values *could* be nullptr for a non-optional tensor if it is empty.
                    // Update the data buffer of the input only if values if non-nullptr.
                    if (values != nullptr) {
                        auto *data_ptr = value.GetMutable<Tensor>()->template MutableData<T>();
                        for (int64_t i = 0; i < values_count; i++) {
                            data_ptr[i] = values[i];
                        }
                    }
                } else {  // "None" Tensor OrtValue. Initialize appropriately.
                    auto ml_tensor = DataTypeImpl::GetType<Tensor>();
                    value.Init(nullptr, ml_tensor, ml_tensor->GetDeleteFunc());
                }

                std::vector<int64_t> dims_for_proto = GetDimsForProto(dims);
                TTypeProto<T> tensor_type_proto(add_shape_to_tensor_data_ ? &dims_for_proto : nullptr);

#if !defined(DISABLE_OPTIONAL_TYPE)
                OptionalTypeProto<T> optional_type_proto(tensor_type_proto.proto);
                auto node_arg = NodeArg(name, !is_optional_type_tensor ? &tensor_type_proto.proto
                                                                       : &optional_type_proto.proto);
#else
                auto node_arg = NodeArg(name, &tensor_type_proto.proto);
#endif

                AddShapeToTensorData(node_arg, dims, dim_params);

                optional<float> rel;
                optional<float> abs;

                if (rel_error != 0.0f) {
                    rel = rel_error;
                }

                if (abs_error != 0.0f) {
                    abs = abs_error;
                }

                data.push_back(
                        Data(std::move(node_arg), std::move(value)));

                // Optional values cannot be initializers
                if (is_initializer && !is_optional_type_tensor) {
                    initializer_index_.push_back(data.size() - 1);
                }
            }
            ORT_CATCH(const std::exception &ex) {
                ORT_HANDLE_EXCEPTION([&]() {
                    std::cerr << "AddData for '" << name << "' threw: " << ex.what();
                });
                ORT_RETHROW;
            }
        }

    private:
        template<typename T>
        void AddSeqData(std::vector<Data> &data, const char *name,
                        const SeqTensors<T> *seq_tensors,
                        bool is_optional_sequence_tensor_type = false) {
#if defined(DISABLE_OPTIONAL_TYPE)
            if (is_optional_sequence_tensor_type) {
          ORT_THROW("Optional type is not supported in this build");
        }
#endif

            std::unique_ptr<TensorSeq> ptr;

            if (seq_tensors) {
                auto num_tensors = seq_tensors->tensors.size();
                std::vector<Tensor> tensors;
                tensors.resize(num_tensors);
                auto elem_type = DataTypeImpl::GetType<T>();
                for (size_t i = 0; i < num_tensors; ++i) {
                    TensorShape shape{seq_tensors->tensors[i].shape};
                    auto values_count = static_cast<int64_t>(seq_tensors->tensors[i].data.size());
                    ORT_ENFORCE(shape.Size() == values_count, values_count,
                                " input values doesn't match tensor size of ", shape.Size());

                    auto allocator = AllocatorManager::Instance().GetAllocator(CPU);
                    auto &tensor = tensors[i];

                    tensor = Tensor(elem_type,
                                    shape,
                                    allocator);

                    auto *data_ptr = tensor.template MutableData<T>();
                    for (int64_t x = 0; x < values_count; ++x) {
                        data_ptr[x] = seq_tensors->tensors[i].data[x];
                    }
                }

                ptr = std::make_unique<TensorSeq>(elem_type);
                ptr->SetElements(std::move(tensors));
            }

            OrtValue value;
            auto mltype = DataTypeImpl::GetType<TensorSeq>();

            // nullptr means None OrtValue which we will skip inserting into the feeds
            value.Init(ptr ? ptr.release() : nullptr, mltype, mltype->GetDeleteFunc());

            SequenceTensorTypeProto<T> sequence_tensor_proto;
#if !defined(DISABLE_OPTIONAL_TYPE)
            OptionalTypeProto<T> optional_type_proto(sequence_tensor_proto.proto);
            auto node_arg = NodeArg(name, !is_optional_sequence_tensor_type
                                          ? &sequence_tensor_proto.proto
                                          : &optional_type_proto.proto);
#else
            auto node_arg = NodeArg(name, &sequence_tensor_proto.proto);
#endif

            data.push_back(Data(std::move(node_arg), std::move(value)));
        }

        std::vector<int64_t> GetDimsForProto(gsl::span<const int64_t> dims);

        void AddShapeToTensorData(NodeArg &node_arg, gsl::span<const int64_t> dims,
                                  const std::vector<std::string> *dim_params);

        void CopyDataToTensor(gsl::span<const gsl::byte> data, Tensor &dst);

#if !defined(DISABLE_SPARSE_TENSORS)

        NodeArg MakeSparseNodeArg(int32_t dtype, const char *name,
                                  const std::vector<int64_t> &dims,
                                  const std::vector<std::string> *dim_params);

        void AddSparseCooTensorData(std::vector<Data> &data,
                                    MLDataType data_type,
                                    const char *name,
                                    const std::vector<int64_t> &dims,
                                    gsl::span<const gsl::byte> values,
                                    gsl::span<const int64_t> indices,
                                    const std::vector<std::string> *dim_params = nullptr);

        void AddSparseCooTensorStrings(std::vector<Data> &data,
                                       const char *name,
                                       const std::vector<int64_t> &dims,
                                       gsl::span<const std::string> values,
                                       gsl::span<const int64_t> indices,
                                       const std::vector<std::string> *dim_params = nullptr);

        void AddSparseCsrTensorData(std::vector<Data> &data,
                                    MLDataType data_type,
                                    const char *name,
                                    const std::vector<int64_t> &dims,
                                    gsl::span<const gsl::byte> values,
                                    gsl::span<const int64_t> inner_indices,
                                    gsl::span<const int64_t> outer_indices,
                                    const std::vector<std::string> *dim_params = nullptr);

        void AddSparseCsrTensorStrings(std::vector<Data> &data,
                                       const char *name,
                                       const std::vector<int64_t> &dims,
                                       gsl::span<const std::string> values,
                                       gsl::span<const int64_t> inner_indices,
                                       gsl::span<const int64_t> outer_indices,
                                       const std::vector<std::string> *dim_params = nullptr);

        void AddSparseTensorData(std::vector<Data> &data, NodeArg node_arg,
                                 std::unique_ptr<SparseTensor> p_tensor);

        void InitOutput();

        TensorShape GetShapeFromShapeProto(const onnx::TensorShapeProto* proto)
        {
            std::vector<int64_t> shape;
            for (int i = 0; i < proto->dim_size(); ++i) {
                shape.push_back(proto->dim(i).dim_value());
            }
            return {shape};
        }

        void AllocOutput(Graph &graph);

        void GraphResolve(Graph& graph, const Graph::ResolveOptions &options, bool cache_enabled)
        {
            Status status = Status::OK();
            if (!cache_enabled) {
                if (add_shape_to_tensor_data_) {
                    //if (add_shape_to_tensor_data_ &&
                    //    expect_result == ExpectResult::kExpectFailure) {
                    // capture possible exceptions from shape inference for invalid testcase
                    ORT_TRY {
                        status = graph.Resolve(options);
                    }
                    ORT_CATCH(const std::exception &ex) {
                        ORT_HANDLE_EXCEPTION([&]() {
                            status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
                        });
                    }
                } else {
                    status = graph.Resolve(options);
                }

//                if (!status.IsOK()) {
//                    if (expect_result == ExpectResult::kExpectFailure) {
//                        EXPECT_TRUE(!status.IsOK());
//                        EXPECT_THAT(status.ErrorMessage(),
//                                    testing::HasSubstr(expected_failure_string));
//                    } else {
//                        LOGS_DEFAULT(ERROR) << "Resolve failed with status: "
//                                            << status.ErrorMessage();
//                        EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
//                    }
//                }


                if (!status.IsOK()) {
                    std::cout << status.ErrorMessage() << std::endl;
                    throw std::runtime_error("status after cache_enabled is not ok");
//                    return;
                }
            }
        }
#endif
    private:
        size_t output_size_ = 0;
        const char *domain_;
        int opset_version_;
        bool add_shape_to_tensor_data_ = true;
        int add_symbolic_dim_to_tensor_data_ = -1;
        int num_run_calls_ = 1;
        std::vector<size_t> initializer_index_;
        std::vector<std::function<void(onnxruntime::Node &node)>> add_attribute_funcs_;

        IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
        std::vector<std::shared_ptr<CustomRegistry>> custom_session_registries_;

        bool verify_output_;

        bool use_determinism_ = false;

        CustomOutputVerifierFn custom_output_verifier_;

        bool add_prepacked_shared_container_to_sessions_ = false;

        onnxruntime::PrepackedWeightsContainer prepacked_weights_container_;
    };
}