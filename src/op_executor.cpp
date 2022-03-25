#include "op_executor.h"
#include "environment.h"
#include "default_providers.h"
#include <core/graph/model_load_utils.h>
#include <core/session/inference_session.h>
#include "onnx/shape_inference/implementation.h"
#include <algorithm>

namespace ortki {
#ifndef NDEBUG
#define DEBUG(s) std::cout << s << std::endl;
#else
#define DEBUG(s) ;
#endif

#define CHECK_STATUS_OK(function)                  \
  do {                                              \
    Status _tmp_status = (function);               \
    if(!_tmp_status.IsOK())                        \
    {                                              \
     ORT_THROW(_tmp_status);\
    }\
  } while (false)

#define CHECK_PROVIDER_STATUS_OK(function)                                                         \
  do {                                                                                              \
    Status _tmp_status = function;                                                                 \
    if(!_tmp_status.IsOK()){                                                                                               \
    std::cout << "provider: " << provider_type << ", error: " << _tmp_status;                      \
    }\
  } while (false)

    template<typename T>
    Tensor copy_sort(const Tensor &src, const AllocatorPtr &allocator) {
        Tensor result(src.DataType(), src.Shape(), allocator);
        memcpy(result.MutableDataRaw(), src.DataRaw(), src.SizeInBytes());
        auto dst_span = gsl::make_span(result.MutableData<T>(), result.MutableData<T>() + result.Shape().Size());
        std::sort(dst_span.begin(), dst_span.end());
        return result;
    }

    void OpExecutor::FillFeedsAndOutputNames(
            std::unordered_map<std::string, OrtValue> &feeds,
            std::vector<std::string> &output_names) {
//        for (auto &output: output_data_) {
//            DEBUG(output.def_.Name());
//            output_names.push_back(output.def_.Name());
//        }

        FillFeeds(feeds);
    }

    void OpExecutor::FillFeeds(std::unordered_map<std::string, OrtValue> &feeds) {
        for (size_t i = 0; i < input_data_.size(); ++i) {
            if (std::find(initializer_index_.begin(), initializer_index_.end(), i) ==
                initializer_index_.end() &&
                input_data_[i].def_.Exists() &&
                // We don't include optional type OrtValues of None because this is
                // how we expect users to deal with sending through "None"s as graph inputs
                // (i.e.) don't send them through at all
                input_data_[i].data_.IsAllocated()) {
                feeds[input_data_[i].def_.Name()] = input_data_[i].data_;
            }
        }
    }

    void OpExecutor::AddNodes(
            onnxruntime::Graph &graph,
            std::vector<onnxruntime::NodeArg *> &graph_input_defs,
            std::vector<onnxruntime::NodeArg *> &graph_output_defs,
            std::vector<std::function<void(onnxruntime::Node &node)>> &add_attribute_funcs) {
        // default behavior is to create a single Node for the op being tested, with
        // node inputs/outputs
        // being 1:1 with graph inputs/outputs.
        auto &node = graph.AddNode("node1", op_, op_, graph_input_defs,
                                   graph_output_defs, nullptr, domain_);
        node.SetExecutionProviderType(kCpuExecutionProvider);
        // Add the attributes if any
        for (auto &add_attribute_fn: add_attribute_funcs)
            add_attribute_fn(node);
    }

    std::vector<int64_t> OpExecutor::GetDimsForProto(gsl::span<const int64_t> dims) {
        std::vector<int64_t> dims_for_proto{dims.begin(), dims.end()};
        if (add_symbolic_dim_to_tensor_data_ >= 0 &&
            dims.size() > static_cast<size_t>(add_symbolic_dim_to_tensor_data_)) {
            dims_for_proto[add_symbolic_dim_to_tensor_data_] = -1;
        }
        return dims_for_proto;
    }

    void OpExecutor::AddShapeToTensorData(NodeArg &node_arg, gsl::span<const int64_t> dims,
                                          const std::vector<std::string> *dim_params) {
        if (dim_params && !(dim_params->empty()) && add_shape_to_tensor_data_) {
            // If dim_params presents, configure node_arg's dim value based on dim_params, which supports symbolic dim and dim broadcast.
            const auto &dim_params_data = *dim_params;
            onnx::TensorShapeProto new_shape;

            // currently hard-code the reserved symbolic names.
            // TODO: when the list grows longer, consider move it to a better place.
            const static std::unordered_set<std::string> reserved_symbolic{"batch", "seq"};

            for (size_t i = 0; i < dim_params_data.size(); ++i) {
                if (reserved_symbolic.find(dim_params_data[i]) != reserved_symbolic.end()) {
                    new_shape.add_dim()->set_dim_param(dim_params_data[i]);
                } else {
                    new_shape.add_dim()->set_dim_value(dims[i]);
                }
            }
            node_arg.SetShape(new_shape);
        }
    }

#if !defined(DISABLE_SPARSE_TENSORS)

    static std::unique_ptr<SparseTensor> MakeSparseTensor(MLDataType data_type, const std::vector<int64_t> &dims) {
        TensorShape shape{dims};
        auto allocator = AllocatorManager::Instance().GetAllocator(CPU);
        auto p_tensor = std::make_unique<SparseTensor>(data_type, shape, allocator);
        return p_tensor;
    }

    void OpExecutor::CopyDataToTensor(gsl::span<const gsl::byte> data, Tensor &dst) {
        ORT_ENFORCE(dst.SizeInBytes() >= data.size_bytes(), "Not enough space in the destination tensor");
        memcpy(dst.MutableDataRaw(), data.data(), data.size_bytes());
    }

    NodeArg OpExecutor::MakeSparseNodeArg(int32_t dtype, const char *name,
                                          const std::vector<int64_t> &dims,
                                          const std::vector<std::string> *dim_params) {
        std::vector<int64_t> dims_for_proto = GetDimsForProto(dims);
        TSparseTensorProto type_proto(dtype, add_shape_to_tensor_data_ ? &dims_for_proto : nullptr);
        NodeArg node_arg(name, &type_proto.proto);
        AddShapeToTensorData(node_arg, dims, dim_params);
        return node_arg;
    }

    void OpExecutor::AddSparseTensorData(std::vector<Data> &data, NodeArg node_arg,
                                         std::unique_ptr<SparseTensor> p_tensor) {
        OrtValue value;
        auto ml_type = DataTypeImpl::GetType<SparseTensor>();
        value.Init(p_tensor.release(), ml_type, ml_type->GetDeleteFunc());
        data.push_back(Data(std::move(node_arg), std::move(value)));
    }

    void OpExecutor::AddSparseCooTensorData(std::vector<Data> &data,
                                            MLDataType data_type,
                                            const char *name,
                                            const std::vector<int64_t> &dims,
                                            gsl::span<const gsl::byte> values,
                                            gsl::span<const int64_t> indices,
                                            const std::vector<std::string> *dim_params) {
        const auto elem_size = data_type->Size();
        const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
        const auto nnz = values.size_bytes() / elem_size;
        ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
        ORT_ENFORCE((nnz == indices.size() || 2 * nnz == indices.size()),
                    "Expecting indices to have either nnz or (2 * nnz) length");
        auto p_tensor = MakeSparseTensor(data_type, dims);
        auto mutator = p_tensor->MakeCooData(nnz, indices.size());
        CopyDataToTensor(values, mutator.Values());
        CopyDataToTensor(indices.as_bytes(), mutator.Indices());

        NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
        AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor));
    }

    void OpExecutor::AddSparseCooTensorStrings(std::vector<Data> &data,
                                               const char *name,
                                               const std::vector<int64_t> &dims,
                                               gsl::span<const std::string> values,
                                               gsl::span<const int64_t> indices,
                                               const std::vector<std::string> *dim_params) {
        auto data_type = DataTypeImpl::GetType<std::string>();
        const auto nnz = values.size();
        const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
        ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
        ORT_ENFORCE((nnz == indices.size() || 2 * nnz == indices.size()),
                    "Expecting indices to have either nnz or (2 * nnz) length");
        auto p_tensor = MakeSparseTensor(data_type, dims);
        // linear index is 1-D index, otherwise 2-D index
        auto mutator = p_tensor->MakeCooData(nnz, indices.size());
        auto mutable_values = mutator.Values().MutableDataAsSpan<std::string>();
        ORT_ENFORCE(values.size() == mutable_values.size(), "Must allocate space for values");
        std::copy(values.cbegin(), values.cend(), mutable_values.begin());
        CopyDataToTensor(indices.as_bytes(), mutator.Indices());
        NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
        AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor));
    }

    void OpExecutor::AddSparseCsrTensorData(std::vector<Data> &data,
                                            MLDataType data_type,
                                            const char *name,
                                            const std::vector<int64_t> &dims,
                                            gsl::span<const gsl::byte> values,
                                            gsl::span<const int64_t> inner_indices,
                                            gsl::span<const int64_t> outer_indices,
                                            const std::vector<std::string> *dim_params) {
        const auto elem_size = data_type->Size();
        const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
        const auto nnz = values.size_bytes() / elem_size;
        ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
        ORT_ENFORCE(nnz == inner_indices.size(), "Expecting the same number of inner_indices as nnz");
        auto p_tensor = MakeSparseTensor(data_type, dims);

        auto mutator = p_tensor->MakeCsrData(nnz, inner_indices.size(), outer_indices.size());
        CopyDataToTensor(values, mutator.Values());
        CopyDataToTensor(inner_indices.as_bytes(), mutator.Inner());
        CopyDataToTensor(outer_indices.as_bytes(), mutator.Outer());

        NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
        AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor));
    }

    void OpExecutor::AddSparseCsrTensorStrings(std::vector<Data> &data,
                                               const char *name,
                                               const std::vector<int64_t> &dims,
                                               gsl::span<const std::string> values,
                                               gsl::span<const int64_t> inner_indices,
                                               gsl::span<const int64_t> outer_indices,
                                               const std::vector<std::string> *dim_params) {
        auto data_type = DataTypeImpl::GetType<std::string>();
        const auto nnz = values.size();
        const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();

        ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
        ORT_ENFORCE(nnz == inner_indices.size(), "Expecting the same number of inner_indices as nnz");
        auto p_tensor = MakeSparseTensor(data_type, dims);

        auto mutator = p_tensor->MakeCsrData(nnz, inner_indices.size(), outer_indices.size());
        auto mutable_values = mutator.Values().MutableDataAsSpan<std::string>();
        ORT_ENFORCE(values.size() == mutable_values.size(), "Must allocate space for values");
        std::copy(values.cbegin(), values.cend(), mutable_values.begin());
        CopyDataToTensor(inner_indices.as_bytes(), mutator.Inner());
        CopyDataToTensor(outer_indices.as_bytes(), mutator.Outer());
        NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
        AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor));
    }

#endif  // !defined(DISABLE_SPARSE_TENSORS)

    void OpExecutor::AddInitializers(onnxruntime::Graph &graph) {
        for (auto index: initializer_index_) {
            auto &data = input_data_[index];
            auto &tensor = data.data_.Get<Tensor>();
            ONNX_NAMESPACE::TensorProto tensor_proto;
            // 1. set dimension
            auto &shape = tensor.Shape();
            for (auto &dim: shape.GetDims()) {
                tensor_proto.add_dims(dim);
            }
            // 2. set type
            tensor_proto.set_data_type(
                    data.def_.TypeAsProto()->tensor_type().elem_type());
            // 3. data
            if (data.def_.TypeAsProto()->tensor_type().elem_type() ==
                ONNX_NAMESPACE::TensorProto_DataType_STRING) {
                const std::string *string_data = tensor.Data<std::string>();
                for (auto i = 0; i < shape.Size(); i++) {
                    tensor_proto.add_string_data(string_data[i]);
                }
            } else {
                auto buffer_size = tensor.DataType()->Size() * shape.Size();
                tensor_proto.set_raw_data(tensor.DataRaw(), buffer_size);
            }
            // 4. name
            tensor_proto.set_name(data.def_.Name());
            graph.AddInitializedTensor(tensor_proto);
        }
    }

    std::unique_ptr<onnxruntime::Model> OpExecutor::BuildGraph(
            const std::unordered_map<std::string, int> &extra_domain_to_version,
            bool allow_released_onnx_opset_only) {
        // Generate the input & output def lists
        std::vector<onnxruntime::NodeArg *> node_input_defs;
        std::vector<onnxruntime::NodeArg *> output_defs;

        for (size_t i = 0; i < input_data_.size(); ++i) {
            node_input_defs.push_back(&input_data_[i].def_);
        }

        for (auto &data: output_data_) {
            output_defs.push_back(&data.def_);
        }

        // Create a simple model
        std::unordered_map<std::string, int> domain_to_version(extra_domain_to_version);
        if (domain_to_version.count(domain_) == 0) {
            domain_to_version.insert({domain_, opset_version_});
        } else {
            auto key_val = extra_domain_to_version.find(domain_);

            ORT_ENFORCE(key_val->second <= opset_version_);

            if (key_val->second < opset_version_) {
                domain_to_version[domain_] = opset_version_;
            }
        }

        auto p_model = std::make_unique<onnxruntime::Model>(
                "test", false, ModelMetaData(), PathString(), custom_schema_registries_,
                domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
                DefaultLoggingManager().DefaultLogger(), allow_released_onnx_opset_only);
        onnxruntime::Graph &graph = p_model->MainGraph();
        AddNodes(graph, node_input_defs, output_defs, add_attribute_funcs_);

        // Add Initializer
        AddInitializers(graph);
        return p_model;
    }

    template<class SessionType>
    std::vector<OrtValue> OpExecutor::ExecuteModel(
            Model &model, SessionType &session_object, const RunOptions *run_options,
            const std::unordered_map<std::string, OrtValue> &feeds,
            const std::vector<std::string> &output_names,
            const std::string &provider_type, bool allow_released_onnx_opset_only) {
        std::string s1;
        const bool rc = model.ToProto().SerializeToString(&s1);
        if (!rc) {
            ORT_THROW("Failed to serialize proto to string");
            return {};
        }
        std::stringstream sstr(s1);
        auto status = session_object.Load(sstr, allow_released_onnx_opset_only);
        // EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
        if (!status.IsOK()) {
            ORT_THROW(std::string("Load failed with status: ") + status.ErrorMessage());
            return {};
        }

        status = session_object.Initialize();

        if (!status.IsOK()) {
            ORT_THROW("ExecuteModel Status is Failed" + status.ErrorMessage());
            return {};
        }

        RunOptions default_run_options{};
        default_run_options.run_tag = op_;
        default_run_options.run_log_verbosity_level = 1;

        std::vector<OrtValue> fetches;
        status =
                session_object.Run(run_options ? *run_options : default_run_options,
                                   feeds, output_names, &fetches);


        return fetches;
    }

    std::vector<OrtValue> OpExecutor::Run(
            const RunOptions *run_options,
            std::vector<std::unique_ptr<IExecutionProvider>> *_,
            ExecutionMode execution_mode) {
        SessionOptions so;
        so.use_per_session_threads = false;
        so.session_logid = op_;
        so.session_log_verbosity_level = 1;
        so.execution_mode = execution_mode;
        so.use_deterministic_compute = use_determinism_;
        so.graph_optimization_level = TransformerLevel::Level1;  // 'Default' == off
        Graph::ResolveOptions options = {};
        options.override_types = true;
        std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
        execution_providers.emplace_back(std::move(DefaultCpuExecutionProvider()));
        return Run(so,
                   run_options, &execution_providers, options);
    }

    std::vector<OrtValue> OpExecutor::Run(
            SessionOptions so,  // Take the SessionOptions by value (i.e. make a copy)
            // because we may need to modify it
            const RunOptions *run_options,
            std::vector<std::unique_ptr<IExecutionProvider>> *execution_providers,
            const Graph::ResolveOptions &options,
            /*out*/ size_t *number_of_pre_packed_weights_counter,
            /*out*/ size_t *number_of_shared_pre_packed_weights_counter) {
        // try
//        {
#ifndef NDEBUG
        run_called_ = true;
#endif

        auto allow_released_onnx_opset_only =
                IsAllowReleasedONNXOpsetsOnlySetForThisTest() &&
                model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();


        DEBUG("current op")
        DEBUG(op_)
        InitOutput();
        auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        auto schema = schema_registry->GetSchema(op_, 15);
//        DEBUG("init output")
        fetches_.clear();
        bool cache_enabled = cached_model_ != nullptr;
        auto p_model = !cache_enabled ? BuildGraph({}, allow_released_onnx_opset_only) : cached_model_;
        auto &graph = p_model->MainGraph();

        std::vector<std::string> output_names;

        for (auto &output: output_data_) {
            output_names.push_back(output.def_.Name());
        }

        GraphResolve(graph, options, cache_enabled);

        AllocOutput(graph);

        // Hookup the inputs and outputs
        std::unordered_map<std::string, OrtValue> feeds;
        FillFeedsAndOutputNames(feeds, output_names);


            InferenceSession session_object{so, GetEnvironment()};

            if (add_prepacked_shared_container_to_sessions_) {
                CHECK_STATUS_OK(session_object.AddPrePackedWeightsContainer(&prepacked_weights_container_));
            }

            if (execution_providers->empty()) {
                ORT_THROW("Empty execution providers vector");
            }
//                ASSERT_TRUE(!execution_providers->empty())
//                        << "Empty execution providers vector.";
            std::string provider_types;

            for (auto &entry: *execution_providers) {
                provider_types += entry->Type() + ":";
                CHECK_STATUS_OK(session_object.RegisterExecutionProvider(std::move(entry)));
            }

            fetches_ = ExecuteModel<InferenceSession>(
                    *p_model, session_object,
                    run_options, feeds, output_names, kCpuExecutionProvider, allow_released_onnx_opset_only);
            // After the model has initialized (happens in ExecuteModel),
            // we should be able to tell how many constant initializers were pre-packed
            // and out of these pre-packed ones how many of them used a "cached" version
            // from the shared container.
            // Populate these value if the user has requested this information.
            if (number_of_pre_packed_weights_counter) {
                *number_of_pre_packed_weights_counter =
                        session_object.GetSessionState().GetNumberOfPrepacksCounter();
            }

            if (number_of_shared_pre_packed_weights_counter) {
                *number_of_shared_pre_packed_weights_counter =
                        session_object.GetSessionState().GetUsedSharedPrePackedWeightCounter();
            }
        return fetches_;
    }

    void OpExecutor::InitOutput() {
        auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        auto schema = schema_registry->GetSchema(op_, opset_version_);
        auto out_size = schema->max_output();
        // spec output size
        if (output_size_ != INT32_MAX) {
            out_size = output_size_;
        }
        // used for split and lstm
        if (out_size == INT32_MAX) {
            if (output_size_ == 0) {
                throw std::runtime_error("output size should not be zero, in op" + schema->Name());
            }
            out_size = output_size_;
        }
        for (int i = 0; i < out_size; ++i) {
            auto &&output_name = schema->outputs()[i].GetName();
            auto mltype = DataTypeImpl::GetType<float>();
            auto name = "output" + std::to_string(i);
            output_data_.emplace_back(NodeArg(name, mltype->GetTypeProto()), OrtValue());
        }
    }

    void OpExecutor::AllocOutput(Graph &graph) {
        for (int i = 0; i < output_data_.size(); ++i) {
            auto out_info = graph.GetOutputs()[i];
            auto proto_shape = out_info->Shape();
            output_data_[i].def_ = NodeArg(out_info->Name(), out_info->TypeAsProto());
        }
    }
}