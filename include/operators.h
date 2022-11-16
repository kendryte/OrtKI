//This file is automatically generated from the onnx def files via tools/gen_operators.py.
#include "tensor.h"
#include "operators_patch.h"
ORTKI_API(ortki::OrtKITensor *) ortki_Abs(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Acos(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Acosh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Add(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_And(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_ArgMax(ortki::OrtKITensor * data, int64_t axis, int64_t keepdims, int64_t select_last_index);
ORTKI_API(ortki::OrtKITensor *) ortki_ArgMin(ortki::OrtKITensor * data, int64_t axis, int64_t keepdims, int64_t select_last_index);
ORTKI_API(ortki::OrtKITensor *) ortki_Asin(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Asinh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Atan(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Atanh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_AveragePool(ortki::OrtKITensor * X, const char* auto_pad, int64_t ceil_mode, int64_t count_include_pad, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Bernoulli(ortki::OrtKITensor * input, int64_t dtype, float seed);
ORTKI_API(ortki::OrtKITensor *) ortki_BitShift(ortki::OrtKITensor * X, ortki::OrtKITensor * Y, const char* direction);
ORTKI_API(ortki::OrtKITensor *) ortki_BlackmanWindow(ortki::OrtKITensor * size, int64_t output_datatype, int64_t periodic);
ORTKI_API(ortki::OrtKITensor *) ortki_Cast(ortki::OrtKITensor * input, int64_t to);
ORTKI_API(ortki::OrtKITensor *) ortki_CastLike(ortki::OrtKITensor * input, ortki::OrtKITensor * target_type);
ORTKI_API(ortki::OrtKITensor *) ortki_Ceil(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Celu(ortki::OrtKITensor * X, float alpha);
ORTKI_API(ortki::OrtKITensor *) ortki_Clip(ortki::OrtKITensor * input, ortki::OrtKITensor * min, ortki::OrtKITensor * max);
ORTKI_API(ortki::OrtKITensor *) ortki_Compress(ortki::OrtKITensor * input, ortki::OrtKITensor * condition, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_Concat(ortki::OrtKITensor ** inputs, size_t input_size, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_ConcatFromSequence(ortki::OrtKITensor ** input_sequence, size_t input_size, int64_t axis, int64_t new_axis);
ORTKI_API(ortki::OrtKITensor *) ortki_Conv(ortki::OrtKITensor * X, ortki::OrtKITensor * W, ortki::OrtKITensor * B, const char* auto_pad, int64_t* dilations, size_t dilations_size, int64_t group, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_ConvInteger(ortki::OrtKITensor * x, ortki::OrtKITensor * w, ortki::OrtKITensor * x_zero_point, ortki::OrtKITensor * w_zero_point, const char* auto_pad, int64_t* dilations, size_t dilations_size, int64_t group, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_ConvTranspose(ortki::OrtKITensor * X, ortki::OrtKITensor * W, ortki::OrtKITensor * B, const char* auto_pad, int64_t* dilations, size_t dilations_size, int64_t group, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* output_padding, size_t output_padding_size, int64_t* output_shape, size_t output_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Cos(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Cosh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_CumSum(ortki::OrtKITensor * x, ortki::OrtKITensor * axis, int64_t exclusive, int64_t reverse);
ORTKI_API(ortki::OrtKITensor *) ortki_DFT(ortki::OrtKITensor * input, ortki::OrtKITensor * dft_length, int64_t axis, int64_t inverse, int64_t onesided);
ORTKI_API(ortki::OrtKITensor *) ortki_DepthToSpace(ortki::OrtKITensor * input, int64_t blocksize, const char* mode);
ORTKI_API(ortki::OrtKITensor *) ortki_DequantizeLinear(ortki::OrtKITensor * x, ortki::OrtKITensor * x_scale, ortki::OrtKITensor * x_zero_point, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_Det(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Div(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Dropout(ortki::OrtKITensor * data, ortki::OrtKITensor * ratio, ortki::OrtKITensor * training_mode, int64_t seed);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_DynamicQuantizeLinear(ortki::OrtKITensor * x);
ORTKI_API(ortki::OrtKITensor *) ortki_Einsum(ortki::OrtKITensor ** Inputs, size_t input_size, const char* equation);
ORTKI_API(ortki::OrtKITensor *) ortki_Elu(ortki::OrtKITensor * X, float alpha);
ORTKI_API(ortki::OrtKITensor *) ortki_Equal(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_Erf(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Exp(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Expand(ortki::OrtKITensor * input, ortki::OrtKITensor * shape);
ORTKI_API(ortki::OrtKITensor *) ortki_EyeLike(ortki::OrtKITensor * input, int64_t dtype, int64_t k);
ORTKI_API(ortki::OrtKITensor *) ortki_Flatten(ortki::OrtKITensor * input, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_Floor(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_GRU(ortki::OrtKITensor * X, ortki::OrtKITensor * W, ortki::OrtKITensor * R, ortki::OrtKITensor * B, ortki::OrtKITensor * sequence_lens, ortki::OrtKITensor * initial_h, float* activation_alpha, size_t activation_alpha_size, float* activation_beta, size_t activation_beta_size, const char** activations, size_t activations_size, float clip, const char* direction, int64_t hidden_size, int64_t layout, int64_t linear_before_reset);
ORTKI_API(ortki::OrtKITensor *) ortki_Gather(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_GatherElements(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_GatherND(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, int64_t batch_dims);
ORTKI_API(ortki::OrtKITensor *) ortki_Gemm(ortki::OrtKITensor * A, ortki::OrtKITensor * B, ortki::OrtKITensor * C, float alpha, float beta, int64_t transA, int64_t transB);
ORTKI_API(ortki::OrtKITensor *) ortki_GlobalAveragePool(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_GlobalLpPool(ortki::OrtKITensor * X, int64_t p);
ORTKI_API(ortki::OrtKITensor *) ortki_GlobalMaxPool(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Greater(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_GreaterOrEqual(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_GridSample(ortki::OrtKITensor * X, ortki::OrtKITensor * grid, int64_t align_corners, const char* mode, const char* padding_mode);
ORTKI_API(ortki::OrtKITensor *) ortki_HammingWindow(ortki::OrtKITensor * size, int64_t output_datatype, int64_t periodic);
ORTKI_API(ortki::OrtKITensor *) ortki_HannWindow(ortki::OrtKITensor * size, int64_t output_datatype, int64_t periodic);
ORTKI_API(ortki::OrtKITensor *) ortki_HardSigmoid(ortki::OrtKITensor * X, float alpha, float beta);
ORTKI_API(ortki::OrtKITensor *) ortki_HardSwish(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Hardmax(ortki::OrtKITensor * input, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_Identity(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_InstanceNormalization(ortki::OrtKITensor * input, ortki::OrtKITensor * scale, ortki::OrtKITensor * B, float epsilon);
ORTKI_API(ortki::OrtKITensor *) ortki_IsInf(ortki::OrtKITensor * X, int64_t detect_negative, int64_t detect_positive);
ORTKI_API(ortki::OrtKITensor *) ortki_IsNaN(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_LRN(ortki::OrtKITensor * X, float alpha, float beta, float bias, int64_t size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_LayerNormalization(ortki::OrtKITensor * X, ortki::OrtKITensor * Scale, ortki::OrtKITensor * B, int64_t axis, float epsilon, int64_t stash_type);
ORTKI_API(ortki::OrtKITensor *) ortki_LeakyRelu(ortki::OrtKITensor * X, float alpha);
ORTKI_API(ortki::OrtKITensor *) ortki_Less(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_LessOrEqual(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_Log(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_LogSoftmax(ortki::OrtKITensor * input, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_LpNormalization(ortki::OrtKITensor * input, int64_t axis, int64_t p);
ORTKI_API(ortki::OrtKITensor *) ortki_LpPool(ortki::OrtKITensor * X, const char* auto_pad, int64_t* kernel_shape, size_t kernel_shape_size, int64_t p, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_MatMul(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_MatMulInteger(ortki::OrtKITensor * A, ortki::OrtKITensor * B, ortki::OrtKITensor * a_zero_point, ortki::OrtKITensor * b_zero_point);
ORTKI_API(ortki::OrtKITensor *) ortki_Max(ortki::OrtKITensor ** data_0, size_t input_size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_MaxPool(ortki::OrtKITensor * X, const char* auto_pad, int64_t ceil_mode, int64_t* dilations, size_t dilations_size, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t storage_order, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_MaxRoiPool(ortki::OrtKITensor * X, ortki::OrtKITensor * rois, int64_t* pooled_shape, size_t pooled_shape_size, float spatial_scale);
ORTKI_API(ortki::OrtKITensor *) ortki_MaxUnpool(ortki::OrtKITensor * X, ortki::OrtKITensor * I, ortki::OrtKITensor * output_shape, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Mean(ortki::OrtKITensor ** data_0, size_t input_size);
ORTKI_API(ortki::OrtKITensor *) ortki_MeanVarianceNormalization(ortki::OrtKITensor * X, int64_t* axes, size_t axes_size);
ORTKI_API(ortki::OrtKITensor *) ortki_MelWeightMatrix(ortki::OrtKITensor * num_mel_bins, ortki::OrtKITensor * dft_length, ortki::OrtKITensor * sample_rate, ortki::OrtKITensor * lower_edge_hertz, ortki::OrtKITensor * upper_edge_hertz, int64_t output_datatype);
ORTKI_API(ortki::OrtKITensor *) ortki_Min(ortki::OrtKITensor ** data_0, size_t input_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Mod(ortki::OrtKITensor * A, ortki::OrtKITensor * B, int64_t fmod);
ORTKI_API(ortki::OrtKITensor *) ortki_Mul(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_Multinomial(ortki::OrtKITensor * input, int64_t dtype, int64_t sample_size, float seed);
ORTKI_API(ortki::OrtKITensor *) ortki_Neg(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_NegativeLogLikelihoodLoss(ortki::OrtKITensor * input, ortki::OrtKITensor * target, ortki::OrtKITensor * weight, int64_t ignore_index, const char* reduction);
ORTKI_API(ortki::OrtKITensor *) ortki_NonMaxSuppression(ortki::OrtKITensor * boxes, ortki::OrtKITensor * scores, ortki::OrtKITensor * max_output_boxes_per_class, ortki::OrtKITensor * iou_threshold, ortki::OrtKITensor * score_threshold, int64_t center_point_box);
ORTKI_API(ortki::OrtKITensor *) ortki_NonZero(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Not(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_OneHot(ortki::OrtKITensor * indices, ortki::OrtKITensor * depth, ortki::OrtKITensor * values, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_OptionalGetElement(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_OptionalHasElement(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Or(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_PRelu(ortki::OrtKITensor * X, ortki::OrtKITensor * slope);
ORTKI_API(ortki::OrtKITensor *) ortki_Pad(ortki::OrtKITensor * data, ortki::OrtKITensor * pads, ortki::OrtKITensor * constant_value, const char* mode);
ORTKI_API(ortki::OrtKITensor *) ortki_Pow(ortki::OrtKITensor * X, ortki::OrtKITensor * Y);
ORTKI_API(ortki::OrtKITensor *) ortki_QLinearConv(ortki::OrtKITensor * x, ortki::OrtKITensor * x_scale, ortki::OrtKITensor * x_zero_point, ortki::OrtKITensor * w, ortki::OrtKITensor * w_scale, ortki::OrtKITensor * w_zero_point, ortki::OrtKITensor * y_scale, ortki::OrtKITensor * y_zero_point, ortki::OrtKITensor * B, const char* auto_pad, int64_t* dilations, size_t dilations_size, int64_t group, int64_t* kernel_shape, size_t kernel_shape_size, int64_t* pads, size_t pads_size, int64_t* strides, size_t strides_size);
ORTKI_API(ortki::OrtKITensor *) ortki_QLinearMatMul(ortki::OrtKITensor * a, ortki::OrtKITensor * a_scale, ortki::OrtKITensor * a_zero_point, ortki::OrtKITensor * b, ortki::OrtKITensor * b_scale, ortki::OrtKITensor * b_zero_point, ortki::OrtKITensor * y_scale, ortki::OrtKITensor * y_zero_point);
ORTKI_API(ortki::OrtKITensor *) ortki_QuantizeLinear(ortki::OrtKITensor * x, ortki::OrtKITensor * y_scale, ortki::OrtKITensor * y_zero_point, int64_t axis);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_RNN(ortki::OrtKITensor * X, ortki::OrtKITensor * W, ortki::OrtKITensor * R, ortki::OrtKITensor * B, ortki::OrtKITensor * sequence_lens, ortki::OrtKITensor * initial_h, float* activation_alpha, size_t activation_alpha_size, float* activation_beta, size_t activation_beta_size, const char** activations, size_t activations_size, float clip, const char* direction, int64_t hidden_size, int64_t layout);
ORTKI_API(ortki::OrtKITensor *) ortki_RandomNormal(int64_t dtype, float mean, float scale, float seed, int64_t* shape, size_t shape_size);
ORTKI_API(ortki::OrtKITensor *) ortki_RandomNormalLike(ortki::OrtKITensor * input, int64_t dtype, float mean, float scale, float seed);
ORTKI_API(ortki::OrtKITensor *) ortki_RandomUniform(int64_t dtype, float high, float low, float seed, int64_t* shape, size_t shape_size);
ORTKI_API(ortki::OrtKITensor *) ortki_RandomUniformLike(ortki::OrtKITensor * input, int64_t dtype, float high, float low, float seed);
ORTKI_API(ortki::OrtKITensor *) ortki_Range(ortki::OrtKITensor * start, ortki::OrtKITensor * limit, ortki::OrtKITensor * delta);
ORTKI_API(ortki::OrtKITensor *) ortki_Reciprocal(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceL1(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceL2(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceLogSum(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceLogSumExp(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceMax(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceMean(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceMin(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceProd(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceSum(ortki::OrtKITensor * data, ortki::OrtKITensor * axes, int64_t keepdims, int64_t noop_with_empty_axes);
ORTKI_API(ortki::OrtKITensor *) ortki_ReduceSumSquare(ortki::OrtKITensor * data, int64_t* axes, size_t axes_size, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_Relu(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Reshape(ortki::OrtKITensor * data, ortki::OrtKITensor * shape, int64_t allowzero);
ORTKI_API(ortki::OrtKITensor *) ortki_ReverseSequence(ortki::OrtKITensor * input, ortki::OrtKITensor * sequence_lens, int64_t batch_axis, int64_t time_axis);
ORTKI_API(ortki::OrtKITensor *) ortki_RoiAlign(ortki::OrtKITensor * X, ortki::OrtKITensor * rois, ortki::OrtKITensor * batch_indices, const char* coordinate_transformation_mode, const char* mode, int64_t output_height, int64_t output_width, int64_t sampling_ratio, float spatial_scale);
ORTKI_API(ortki::OrtKITensor *) ortki_Round(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_STFT(ortki::OrtKITensor * signal, ortki::OrtKITensor * frame_step, ortki::OrtKITensor * window, ortki::OrtKITensor * frame_length, int64_t onesided);
ORTKI_API(ortki::OrtKITensor *) ortki_Scatter(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, ortki::OrtKITensor * updates, int64_t axis);
ORTKI_API(ortki::OrtKITensor *) ortki_ScatterElements(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, ortki::OrtKITensor * updates, int64_t axis, const char* reduction);
ORTKI_API(ortki::OrtKITensor *) ortki_ScatterND(ortki::OrtKITensor * data, ortki::OrtKITensor * indices, ortki::OrtKITensor * updates, const char* reduction);
ORTKI_API(ortki::OrtKITensor *) ortki_Selu(ortki::OrtKITensor * X, float alpha, float gamma);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceAt(ortki::OrtKITensor ** input_sequence, size_t input_size, ortki::OrtKITensor * position);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceConstruct(ortki::OrtKITensor ** inputs, size_t input_size);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceEmpty(int64_t dtype);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceErase(ortki::OrtKITensor ** input_sequence, size_t input_size, ortki::OrtKITensor * position);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceInsert(ortki::OrtKITensor ** input_sequence, size_t input_size, ortki::OrtKITensor * tensor, ortki::OrtKITensor * position);
ORTKI_API(ortki::OrtKITensor *) ortki_SequenceLength(ortki::OrtKITensor ** input_sequence, size_t input_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Shape(ortki::OrtKITensor * data, int64_t end, int64_t start);
ORTKI_API(ortki::OrtKITensor *) ortki_Shrink(ortki::OrtKITensor * input, float bias, float lambd);
ORTKI_API(ortki::OrtKITensor *) ortki_Sigmoid(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Sign(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Sin(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Sinh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Size(ortki::OrtKITensor * data);
ORTKI_API(ortki::OrtKITensor *) ortki_Slice(ortki::OrtKITensor * data, ortki::OrtKITensor * starts, ortki::OrtKITensor * ends, ortki::OrtKITensor * axes, ortki::OrtKITensor * steps);
ORTKI_API(ortki::OrtKITensor *) ortki_Softmax(ortki::OrtKITensor * input, int64_t axis);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_SoftmaxCrossEntropyLoss(ortki::OrtKITensor * scores, ortki::OrtKITensor * labels, ortki::OrtKITensor * weights, int64_t ignore_index, const char* reduction);
ORTKI_API(ortki::OrtKITensor *) ortki_Softplus(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Softsign(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_SpaceToDepth(ortki::OrtKITensor * input, int64_t blocksize);
ORTKI_API(ortki::OrtKITensor *) ortki_SplitToSequence(ortki::OrtKITensor * input, ortki::OrtKITensor * split, int64_t axis, int64_t keepdims);
ORTKI_API(ortki::OrtKITensor *) ortki_Sqrt(ortki::OrtKITensor * X);
ORTKI_API(ortki::OrtKITensor *) ortki_Squeeze(ortki::OrtKITensor * data, ortki::OrtKITensor * axes);
ORTKI_API(ortki::OrtKITensor *) ortki_StringNormalizer(ortki::OrtKITensor * X, const char* case_change_action, int64_t is_case_sensitive, const char* locale, const char** stopwords, size_t stopwords_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Sub(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_Sum(ortki::OrtKITensor ** data_0, size_t input_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Tan(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_Tanh(ortki::OrtKITensor * input);
ORTKI_API(ortki::OrtKITensor *) ortki_TfIdfVectorizer(ortki::OrtKITensor * X, int64_t max_gram_length, int64_t max_skip_count, int64_t min_gram_length, const char* mode, int64_t* ngram_counts, size_t ngram_counts_size, int64_t* ngram_indexes, size_t ngram_indexes_size, int64_t* pool_int64s, size_t pool_int64s_size, const char** pool_strings, size_t pool_strings_size, float* weights, size_t weights_size);
ORTKI_API(ortki::OrtKITensor *) ortki_ThresholdedRelu(ortki::OrtKITensor * X, float alpha);
ORTKI_API(ortki::OrtKITensor *) ortki_Tile(ortki::OrtKITensor * input, ortki::OrtKITensor * repeats);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_TopK(ortki::OrtKITensor * X, ortki::OrtKITensor * K, int64_t axis, int64_t largest, int64_t sorted);
ORTKI_API(ortki::OrtKITensor *) ortki_Transpose(ortki::OrtKITensor * data, int64_t* perm, size_t perm_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Trilu(ortki::OrtKITensor * input, ortki::OrtKITensor * k, int64_t upper);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Unique(ortki::OrtKITensor * X, int64_t axis, int64_t sorted);
ORTKI_API(ortki::OrtKITensor *) ortki_Unsqueeze(ortki::OrtKITensor * data, ortki::OrtKITensor * axes);
ORTKI_API(ortki::OrtKITensor *) ortki_Where(ortki::OrtKITensor * condition, ortki::OrtKITensor * X, ortki::OrtKITensor * Y);
ORTKI_API(ortki::OrtKITensor *) ortki_Xor(ortki::OrtKITensor * A, ortki::OrtKITensor * B);
ORTKI_API(ortki::OrtKITensor *) ortki_ArrayFeatureExtractor(ortki::OrtKITensor * X, ortki::OrtKITensor * Y);
ORTKI_API(ortki::OrtKITensor *) ortki_Binarizer(ortki::OrtKITensor * X, float threshold);
ORTKI_API(ortki::OrtKITensor *) ortki_CastMap(ortki::OrtKITensor * X, const char* cast_to, const char* map_form, int64_t max_map);
ORTKI_API(ortki::OrtKITensor *) ortki_CategoryMapper(ortki::OrtKITensor * X, int64_t* cats_int64s, size_t cats_int64s_size, const char** cats_strings, size_t cats_strings_size, int64_t default_int64, const char* default_string);
ORTKI_API(ortki::OrtKITensor *) ortki_DictVectorizer(ortki::OrtKITensor * X, int64_t* int64_vocabulary, size_t int64_vocabulary_size, const char** string_vocabulary, size_t string_vocabulary_size);
ORTKI_API(ortki::OrtKITensor *) ortki_FeatureVectorizer(ortki::OrtKITensor ** X, size_t input_size, int64_t* inputdimensions, size_t inputdimensions_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Imputer(ortki::OrtKITensor * X, float* imputed_value_floats, size_t imputed_value_floats_size, int64_t* imputed_value_int64s, size_t imputed_value_int64s_size, float replaced_value_float, int64_t replaced_value_int64);
ORTKI_API(ortki::OrtKITensor *) ortki_LabelEncoder(ortki::OrtKITensor * X, float default_float, int64_t default_int64, const char* default_string, float* keys_floats, size_t keys_floats_size, int64_t* keys_int64s, size_t keys_int64s_size, const char** keys_strings, size_t keys_strings_size, float* values_floats, size_t values_floats_size, int64_t* values_int64s, size_t values_int64s_size, const char** values_strings, size_t values_strings_size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_LinearClassifier(ortki::OrtKITensor * X, int64_t* classlabels_ints, size_t classlabels_ints_size, const char** classlabels_strings, size_t classlabels_strings_size, float* coefficients, size_t coefficients_size, float* intercepts, size_t intercepts_size, int64_t multi_class, const char* post_transform);
ORTKI_API(ortki::OrtKITensor *) ortki_LinearRegressor(ortki::OrtKITensor * X, float* coefficients, size_t coefficients_size, float* intercepts, size_t intercepts_size, const char* post_transform, int64_t targets);
ORTKI_API(ortki::OrtKITensor *) ortki_Normalizer(ortki::OrtKITensor * X, const char* norm);
ORTKI_API(ortki::OrtKITensor *) ortki_OneHotEncoder(ortki::OrtKITensor * X, int64_t* cats_int64s, size_t cats_int64s_size, const char** cats_strings, size_t cats_strings_size, int64_t zeros);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_SVMClassifier(ortki::OrtKITensor * X, int64_t* classlabels_ints, size_t classlabels_ints_size, const char** classlabels_strings, size_t classlabels_strings_size, float* coefficients, size_t coefficients_size, float* kernel_params, size_t kernel_params_size, const char* kernel_type, const char* post_transform, float* prob_a, size_t prob_a_size, float* prob_b, size_t prob_b_size, float* rho, size_t rho_size, float* support_vectors, size_t support_vectors_size, int64_t* vectors_per_class, size_t vectors_per_class_size);
ORTKI_API(ortki::OrtKITensor *) ortki_SVMRegressor(ortki::OrtKITensor * X, float* coefficients, size_t coefficients_size, float* kernel_params, size_t kernel_params_size, const char* kernel_type, int64_t n_supports, int64_t one_class, const char* post_transform, float* rho, size_t rho_size, float* support_vectors, size_t support_vectors_size);
ORTKI_API(ortki::OrtKITensor *) ortki_Scaler(ortki::OrtKITensor * X, float* offset, size_t offset_size, float* scale, size_t scale_size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_TreeEnsembleClassifier(ortki::OrtKITensor * X, float* base_values, size_t base_values_size, ortki::OrtKITensor * base_values_as_tensor, int64_t* class_ids, size_t class_ids_size, int64_t* class_nodeids, size_t class_nodeids_size, int64_t* class_treeids, size_t class_treeids_size, float* class_weights, size_t class_weights_size, ortki::OrtKITensor * class_weights_as_tensor, int64_t* classlabels_int64s, size_t classlabels_int64s_size, const char** classlabels_strings, size_t classlabels_strings_size, int64_t* nodes_falsenodeids, size_t nodes_falsenodeids_size, int64_t* nodes_featureids, size_t nodes_featureids_size, float* nodes_hitrates, size_t nodes_hitrates_size, ortki::OrtKITensor * nodes_hitrates_as_tensor, int64_t* nodes_missing_value_tracks_true, size_t nodes_missing_value_tracks_true_size, const char** nodes_modes, size_t nodes_modes_size, int64_t* nodes_nodeids, size_t nodes_nodeids_size, int64_t* nodes_treeids, size_t nodes_treeids_size, int64_t* nodes_truenodeids, size_t nodes_truenodeids_size, float* nodes_values, size_t nodes_values_size, ortki::OrtKITensor * nodes_values_as_tensor, const char* post_transform);
ORTKI_API(ortki::OrtKITensor *) ortki_TreeEnsembleRegressor(ortki::OrtKITensor * X, const char* aggregate_function, float* base_values, size_t base_values_size, ortki::OrtKITensor * base_values_as_tensor, int64_t n_targets, int64_t* nodes_falsenodeids, size_t nodes_falsenodeids_size, int64_t* nodes_featureids, size_t nodes_featureids_size, float* nodes_hitrates, size_t nodes_hitrates_size, ortki::OrtKITensor * nodes_hitrates_as_tensor, int64_t* nodes_missing_value_tracks_true, size_t nodes_missing_value_tracks_true_size, const char** nodes_modes, size_t nodes_modes_size, int64_t* nodes_nodeids, size_t nodes_nodeids_size, int64_t* nodes_treeids, size_t nodes_treeids_size, int64_t* nodes_truenodeids, size_t nodes_truenodeids_size, float* nodes_values, size_t nodes_values_size, ortki::OrtKITensor * nodes_values_as_tensor, const char* post_transform, int64_t* target_ids, size_t target_ids_size, int64_t* target_nodeids, size_t target_nodeids_size, int64_t* target_treeids, size_t target_treeids_size, float* target_weights, size_t target_weights_size, ortki::OrtKITensor * target_weights_as_tensor);
ORTKI_API(ortki::OrtKITensor *) ortki_ZipMap(ortki::OrtKITensor * X, int64_t* classlabels_int64s, size_t classlabels_int64s_size, const char** classlabels_strings, size_t classlabels_strings_size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Adagrad(ortki::OrtKITensor * R, ortki::OrtKITensor * T, ortki::OrtKITensor * inputs, float decay_factor, float epsilon, float norm_coefficient);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Adam(ortki::OrtKITensor * R, ortki::OrtKITensor * T, ortki::OrtKITensor * inputs, float alpha, float beta, float epsilon, float norm_coefficient, float norm_coefficient_post);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Gradient(ortki::OrtKITensor ** Inputs, size_t input_size, const char** xs, size_t xs_size, const char* y, const char** zs, size_t zs_size);
ORTKI_API(ortki::OrtKITensorSeq *) ortki_Momentum(ortki::OrtKITensor * R, ortki::OrtKITensor * T, ortki::OrtKITensor * inputs, float alpha, float beta, const char* mode, float norm_coefficient);