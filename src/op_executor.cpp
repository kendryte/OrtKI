#include "op_executor.h"
#include "environment.h"
#include "default_providers.h"
#include <core/graph/model_load_utils.h>
#include <core/session/inference_session.h>
#include "onnx/shape_inference/implementation.h"
#include <algorithm>

namespace ortki {
#define EXPECT_TRUE(a) \
    std::cout
#define ASSERT_TRUE(a) \
    std::cout

#define CHECK_STATUS_OK(function)                  \
  do {                                              \
    Status _tmp_status = (function);                \
    ASSERT_TRUE(_tmp_status.IsOK()) << _tmp_status; \
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
        for (auto &output: output_data_) {
            if (output.def_.Exists())
                output_names.push_back(output.def_.Name());
        }

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
            LOGS_DEFAULT(ERROR) << "Failed to serialize proto to string";
            return {};
        }
        std::stringstream sstr(s1);
        auto status = session_object.Load(sstr, allow_released_onnx_opset_only);
        // EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
        if (!status.IsOK()) {
            LOGS_DEFAULT(ERROR) << "Load failed with status: " << status.ErrorMessage();
            return {};
        }

        status = session_object.Initialize();

//        if (!status.IsOK()) {
//            if (expect_result == ExpectResult::kExpectFailure) {
//                EXPECT_TRUE(!status.IsOK());
//                // Disable expected_failure_string checks for OpenVINO EP
//                if (provider_type != kOpenVINOExecutionProvider) {
//                    EXPECT_THAT(status.ErrorMessage(),
//                                testing::HasSubstr(expected_failure_string));
//                }
//            } else {
//                LOGS_DEFAULT(ERROR) << "Initialize failed with status: "
//                                    << status.ErrorMessage();
//                EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
//            }
//        }

        if (!status.IsOK()) {
            return {};
        }

        RunOptions default_run_options{};
        default_run_options.run_tag = op_;
        default_run_options.run_log_verbosity_level = 1;

        std::vector<OrtValue> fetches;
        for (int i = 0; i < num_run_calls_; ++i) {
            fetches.clear();
            status =
                    session_object.Run(run_options ? *run_options : default_run_options,
                                       feeds, output_names, &fetches);

//            if (status.IsOK()) {
//                return fetches;
////                if (expect_result == ExpectResult::kExpectFailure) {
////                    return {};
////                }
//            } else {
////                if (expect_result == ExpectResult::kExpectFailure) {
////                    // Disable expected_failure_string checks for MKL-DNN and OpenVINO EP's
////                    if (provider_type != kDnnlExecutionProvider &&
////                        provider_type != kOpenVINOExecutionProvider) {
////                        EXPECT_THAT(status.ErrorMessage(),
////                                    testing::HasSubstr(expected_failure_string));
////                    }
////                } else {
////                    LOGS_DEFAULT(ERROR) << "Run failed with status: "
////                                        << status.ErrorMessage();
////                    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
////                }
//                return {};
//            }
        }

        // Verify the outputs
        // Todo: support check output with map/sequence/....
//        if (verify_output_) {
//            if (custom_output_verifier_) {
//                // do custom verification if provided
//                custom_output_verifier_(fetches, provider_type);
//            } else {
//                // default verification
//                size_t idx = 0;
//                for (auto &expected_data: output_data_) {
//                    OrtValue &ort_value = fetches[idx];
//                    if (ort_value.Fence())
//                        ort_value.Fence()->BeforeUsingAsInput(
//                                onnxruntime::kCpuExecutionProvider, 0);
//
//                    if (expected_data.def_.Exists()) {           // optional edges won't exist (so skip them)
//                        if (!expected_data.data_.IsAllocated()) {  // optional type output (None)
//                            EXPECT_TRUE(!ort_value.IsAllocated())
//                                    << "Expected to see an output of None "
//                                    << "but instead got an output that wasn't None";
//
//                            // Make sure types align
//                            EXPECT_EQ(expected_data.data_.Type(), ort_value.Type())
//                                    << "Expected optional type: " << expected_data.data_.Type()
//                                    << " but instead got optional type: " << ort_value.Type();
//                        } else if (expected_data.data_.IsTensor()) {
//                            // verify output shape inference when input defs have shape
//                            if (add_shape_to_tensor_data_) {
//                                auto out_shape_proto = expected_data.def_.Shape();
//                                EXPECT_TRUE(out_shape_proto != nullptr);
//                                const auto &tensor_shape =
//                                        utils::GetTensorShapeFromTensorShapeProto(*out_shape_proto);
//                                const auto &inferred_dims = tensor_shape.GetDims();
//                                const auto &expected_shape =
//                                        expected_data.data_.Get<Tensor>().Shape();
//                                EXPECT_TRUE(inferred_dims.size() ==
//                                            expected_shape.NumDimensions());
//                                for (size_t d = 0; d < inferred_dims.size(); ++d) {
//                                    // check equal unless the input involved a symbolic dimension
//                                    if (inferred_dims[d] != -1) {
//                                        EXPECT_EQ(expected_shape[d], inferred_dims[d])
//                                                << "Output idx = " << idx << " dim = " << d;
//                                    }
//                                }
//                            }
//
//                            Check(expected_data, ort_value.Get<Tensor>(), provider_type);
//                        } else {
//                            Check(expected_data, ort_value, provider_type);
//                        }
//
//                        ++idx;
//
//                        // skip missing trailing optional outputs
//                        if (idx == fetches.size())
//                            break;
//                    }
//                }
//            }
//        }

        return fetches;
    }

    std::vector<OrtValue> OpExecutor::Run(
            const std::unordered_set<std::string> &excluded_provider_types,
            const RunOptions *run_options,
            std::vector<std::unique_ptr<IExecutionProvider>> *execution_providers,
            ExecutionMode execution_mode) {
        SessionOptions so;
        so.use_per_session_threads = false;
        so.session_logid = op_;
        so.session_log_verbosity_level = 1;
        so.execution_mode = execution_mode;
        so.use_deterministic_compute = use_determinism_;
        so.graph_optimization_level = TransformerLevel::Default;  // 'Default' == off
        Graph::ResolveOptions options = {};
        options.override_types = true;
        return Run(so, excluded_provider_types,
            run_options, execution_providers, options);
    }

    std::vector<OrtValue> OpExecutor::Run(
            SessionOptions so,  // Take the SessionOptions by value (i.e. make a copy)
            // because we may need to modify it
            const std::unordered_set<std::string> &excluded_provider_types,
            const RunOptions *run_options,
            std::vector<std::unique_ptr<IExecutionProvider>> *execution_providers,
            const Graph::ResolveOptions &options,
            /*out*/ size_t *number_of_pre_packed_weights_counter,
            /*out*/ size_t *number_of_shared_pre_packed_weights_counter) {
        std::string cur_provider = "not set";
        // try
        {
#ifndef NDEBUG
            run_called_ = true;
#endif


            // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
            // `IsAllowReleasedONNXOpsetsOnlySetForThisTest()` is for this specific OpExecutor instance
            // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `IsAllowReleasedONNXOpsetsOnlySetForThisTest()`
            // are both true
            auto allow_released_onnx_opset_only =
                    IsAllowReleasedONNXOpsetsOnlySetForThisTest() &&
                    model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

            if (allow_released_onnx_opset_only) {
                auto &onnx_released_versions =
                        ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().LastReleaseVersionMap();
                auto it = onnx_released_versions.find(domain_);
                if (it != onnx_released_versions.end() && opset_version_ > it->second) {
                    LOGS_DEFAULT(WARNING)
                        << "Encountered model with opset version greater than released onnx opset version. "
                        << "Skipping this test. To run this test set environment variable ALLOW_RELEASED_ONNX_OPSET_ONLY to \"0\". "
                        << "Opset version of current model is " << opset_version_
                        << ", the latest released onnx opset version is " << it->second << ".";
                    // GTEST_SKIP();
                    throw std::runtime_error("opset version error");
                }
            }


            InitOutput();
            auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
            auto schema = schema_registry->GetSchema("Add", 15);
            auto max_input = schema->max_input();
            auto max_output = schema->max_output();

//            std::cout << "add output" << std::endl;
            // AddOutput("C", new OrtKITensor(new int(), onnx::TensorProto_DataType_INT32, std::vector<int64_t>{}));
            fetches_.clear();
            bool cache_enabled = cached_model_ != nullptr;
            auto p_model = !cache_enabled ? BuildGraph({}, allow_released_onnx_opset_only) : cached_model_;
            auto &graph = p_model->MainGraph();

//            std::cout << "graph resolve" << std::endl;
            GraphResolve(graph, options, cache_enabled);

//            std::cout << "add output" << std::endl;
            AllocOutput(graph);

//            graph.SetGraphProtoSyncNeeded();
//            graph.SetGraphResolveNeeded();
//
//            GraphResolve(graph, options, cache_enabled);

            // Hookup the inputs and outputs
            std::unordered_map<std::string, OrtValue> feeds;
            std::vector<std::string> output_names;
            FillFeedsAndOutputNames(feeds, output_names);
            // Run the model
            static const std::string all_provider_types[] = {
                    kCpuExecutionProvider,
//                    kCudaExecutionProvider,
//                    kDnnlExecutionProvider,
//                    kNupharExecutionProvider,
//                    kTensorrtExecutionProvider,
//                    kOpenVINOExecutionProvider,
//                    kDmlExecutionProvider,
//                    kAclExecutionProvider,
//                    kArmNNExecutionProvider,
//                    kNnapiExecutionProvider,
//                    kRocmExecutionProvider,
//                    kCoreMLExecutionProvider,
            };

            bool has_run = false;

            if (execution_providers) {
                for (auto &entry: *execution_providers) {
                    if (entry->Type() == kDmlExecutionProvider) {
                        so.enable_mem_pattern = false;
                        so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
                        break;
                    }
                }

                InferenceSession session_object{so, GetEnvironment()};

                if (add_prepacked_shared_container_to_sessions_) {
                    CHECK_STATUS_OK(session_object.AddPrePackedWeightsContainer(&prepacked_weights_container_));
                }

                if(execution_providers->empty())
                {
                    ORT_THROW("Empty execution providers vector");
                }
//                ASSERT_TRUE(!execution_providers->empty())
//                        << "Empty execution providers vector.";
                std::string provider_types;

                for (auto &entry: *execution_providers) {
                    provider_types += entry->Type() + ":";
                    CHECK_STATUS_OK(session_object.RegisterExecutionProvider(std::move(entry)));
                }

//                std::cout << "Execute" << std::endl;
                fetches_ = ExecuteModel<InferenceSession>(
                        *p_model, session_object,
                        run_options, feeds, output_names, provider_types, allow_released_onnx_opset_only);

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

            } else {
                for (const std::string &provider_type: all_provider_types) {
                    if (excluded_provider_types.count(provider_type) > 0)
                        continue;

                    cur_provider = provider_type;

                    if (provider_type == kDmlExecutionProvider) {
                        so.enable_mem_pattern = false;
                        so.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
                    }
                    InferenceSession session_object{so, GetEnvironment()};

                    if (add_prepacked_shared_container_to_sessions_) {
                        CHECK_STATUS_OK(session_object.AddPrePackedWeightsContainer(&prepacked_weights_container_));
                    }

                    for (auto &custom_session_registry: custom_session_registries_)
                        CHECK_PROVIDER_STATUS_OK(session_object.RegisterCustomRegistry(custom_session_registry));

                    std::unique_ptr<IExecutionProvider> execution_provider = DefaultCpuExecutionProvider();
//                    if (provider_type == onnxruntime::kCpuExecutionProvider)
//                        execution_provider = DefaultCpuExecutionProvider();
//                    else if (provider_type == onnxruntime::kCudaExecutionProvider)
//                        execution_provider = DefaultCudaExecutionProvider();
//                    else if (provider_type == onnxruntime::kDnnlExecutionProvider)
//                        execution_provider = DefaultDnnlExecutionProvider();
//                    else if (provider_type == onnxruntime::kOpenVINOExecutionProvider)
//                        execution_provider = DefaultOpenVINOExecutionProvider();
//                    else if (provider_type == onnxruntime::kNupharExecutionProvider)
//                        execution_provider = DefaultNupharExecutionProvider();
//                    else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
//                        execution_provider = DefaultTensorrtExecutionProvider();
//                    else if (provider_type == onnxruntime::kNnapiExecutionProvider)
//                        execution_provider = DefaultNnapiExecutionProvider();
//                    else if (provider_type == onnxruntime::kRknpuExecutionProvider)
//                        execution_provider = DefaultRknpuExecutionProvider();
//                    else if (provider_type == onnxruntime::kAclExecutionProvider)
//                        execution_provider = DefaultAclExecutionProvider();
//                    else if (provider_type == onnxruntime::kArmNNExecutionProvider)
//                        execution_provider = DefaultArmNNExecutionProvider();
//                    else if (provider_type == onnxruntime::kRocmExecutionProvider)
//                        execution_provider = DefaultRocmExecutionProvider();
//                    else if (provider_type == onnxruntime::kCoreMLExecutionProvider)
//                        execution_provider = DefaultCoreMLExecutionProvider();
//                    // skip if execution provider is disabled
//                    if (execution_provider == nullptr)
//                        continue;

                    bool valid = true;

                    // set execution provider for all nodes in the graph
                    for (auto &node: graph.Nodes()) {
                        if (node.OpType() == kConstant)
                            continue;

                        // if node is not registered for the provider, skip
                        node.SetExecutionProviderType(provider_type);
//                        if (provider_type == onnxruntime::kOpenVINOExecutionProvider ||
//                            provider_type == onnxruntime::kTensorrtExecutionProvider ||
//                            provider_type == onnxruntime::kNupharExecutionProvider ||
//                            // provider_type == onnxruntime::kStvmExecutionProvider ||
//                            provider_type == onnxruntime::kNnapiExecutionProvider ||
//                            provider_type == onnxruntime::kCoreMLExecutionProvider ||
//                            provider_type == onnxruntime::kDnnlExecutionProvider)
//                            continue;
                        auto reg = execution_provider->GetKernelRegistry();
                        if (!KernelRegistry::HasImplementationOf(*reg, node, execution_provider->Type())) {
                            valid = false;
                            for (auto &custom_session_registry: custom_session_registries_) {
                                if (KernelRegistry::HasImplementationOf(*custom_session_registry->GetKernelRegistry(),
                                                                        node, execution_provider->Type())) {
                                    valid = true;
                                    break;
                                }
                            }

                            if (!valid) {
                                break;
                            }
                        }
                    }

                    if (!valid)
                        continue;

                    for (auto &custom_session_registry: custom_session_registries_)
                        CHECK_PROVIDER_STATUS_OK(session_object.RegisterCustomRegistry(custom_session_registry));

                    has_run = true;

                    CHECK_PROVIDER_STATUS_OK(session_object.RegisterExecutionProvider(std::move(execution_provider)));
                    fetches_ = ExecuteModel<InferenceSession>(
                            *p_model, session_object,
                            run_options, feeds, output_names, provider_type, allow_released_onnx_opset_only);
//                    std::cout << "Execute" << std::endl;
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

                    cur_provider = "not set";
                }

                if(!has_run)
                {
                    std::cout << "No registered execution providers were able to run.";
                }
//                EXPECT_TRUE(has_run)
//                        << "No registered execution providers were able to run.";
            }
            // p_model->MainGraph().GetOutputs()
        }
//        ORT_CATCH(const std::exception &ex) {
//            ORT_HANDLE_EXCEPTION([&]() {
//                std::cerr << ex.what() << "\nProvider:" << cur_provider << "\n";
//            });
//            // rethrow as some tests for error handling expect this
//            ORT_RETHROW;
//        }
        return fetches_;
    }

    void OpExecutor::InitOutput()
    {
        auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
        auto schema = schema_registry->GetSchema(op_, opset_version_);
        auto out_size = schema->max_output();
        // used for split
        if(out_size == INT32_MAX)
        {
            if(output_size_ == 0)
            {
                throw std::runtime_error("output size should not be zero, in op" + schema->Name());
            }
            out_size = output_size_;
        }
        for (int i = 0; i < out_size; ++i) {
            auto &&output_name = schema->outputs()[i].GetName();
            auto mltype = DataTypeImpl::GetType<float>();
            output_data_.emplace_back(NodeArg("node" + std::to_string(i), mltype->GetTypeProto()), OrtValue());
        }
    }

    void OpExecutor::AllocOutput(Graph &graph) {
        for (int i = 0; i < output_data_.size(); ++i) {
            auto out_info = graph.GetOutputs()[0];
            auto proto_shape = out_info->Shape();
            output_data_[i].def_ = NodeArg(out_info->Name(), out_info->TypeAsProto());
            // output_data_[i].def_.SetShape(*proto_shape);

//                auto shape = GetShapeFromShapeProto(proto_shape);
//                auto *buffer = new int[shape.Size()];
//                auto *tensor = new onnxruntime::Tensor(GetDataType(out_info->TypeAsProto()), shape,
//                                                      reinterpret_cast<void*>(buffer), OrtMemoryInfo());
//                output_data_[i].data_.Init(tensor, onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(), [](auto&&){});
        }
    }

//    void OpExecutor::AddReferenceOutputs(const std::string &model_path) {
//        SessionOptions so;
//        so.session_logid = op_;
//        so.session_log_verbosity_level = 1;
//        so.graph_optimization_level = TransformerLevel::Default;
//
//        RunOptions run_options;
//        run_options.run_tag = op_;
//        run_options.run_log_verbosity_level = 1;
//
//        Status status;
//        InferenceSession subgraph_session_object{so, GetEnvironment()};
//        ASSERT_TRUE((status = subgraph_session_object.Load(model_path)).IsOK()) << status;
//        ASSERT_TRUE((status = subgraph_session_object.Initialize()).IsOK()) << status;
//
//        // Retrieve output names
//        auto model_outputs = subgraph_session_object.GetModelOutputs();
//        ASSERT_TRUE(model_outputs.first.IsOK());
//        std::vector<std::string> output_names;
//        std::transform(model_outputs.second->begin(),
//                       model_outputs.second->end(),
//                       std::back_inserter(output_names),
//                       [](const onnxruntime::NodeArg *node_arg) -> std::string { return node_arg->Name(); });
//
//        NameMLValMap feeds;
//        for (size_t i = 0; i < input_data_.size(); ++i) {
//            if (input_data_[i].def_.Exists()) {
//                feeds[input_data_[i].def_.Name()] = input_data_[i].data_;
//            }
//        }
//
//        std::vector<OrtValue> subgraph_fetches;
//        ASSERT_TRUE((status = subgraph_session_object.Run(run_options, feeds, output_names, &subgraph_fetches)).IsOK())
//                << status;
//
//        for (size_t out_idx = 0; out_idx < subgraph_fetches.size(); out_idx++) {
//            // Retrieve TypeProto
//            ASSERT_TRUE(subgraph_fetches[out_idx].Type()->IsTensorType()) << status;
//            const Tensor &t = subgraph_fetches[out_idx].Get<Tensor>();
//            const TensorTypeBase *tensor_type = DataTypeImpl::TensorTypeFromONNXEnum(t.GetElementType());
//
//            // Construct a temp TypeProto with shape information
//            ONNX_NAMESPACE::TypeProto tmp_type_proto(*(tensor_type->GetTypeProto()));
//            auto mutable_shape = tmp_type_proto.mutable_tensor_type()->mutable_shape();
//            for (auto i: t.Shape().GetDims()) {
//                auto *mutable_dim = mutable_shape->add_dim();
//                mutable_dim->set_dim_value(i);
//            }
//
//            output_data_.push_back(Data(NodeArg(output_names[out_idx], &tmp_type_proto),
//                                        std::move(subgraph_fetches[out_idx])));
//        }
//    }

//#ifdef ENABLE_TRAINING
//    template std::vector<OrtValue> OpExecutor::ExecuteModel<training::TrainingSession>(
//        Model& model, training::TrainingSession& session_object,
//        ExpectResult expect_result, const std::string& expected_failure_string,
//        const RunOptions* run_options,
//        const std::unordered_map<std::string, OrtValue>& feeds,
//        const std::vector<std::string>& output_names, const std::string& provider_type,
//        bool allow_released_onnx_opset_only);
//#endif
}