#include <operators_patch.h>
#include <op_executor.h>

template<typename T = int>
ortki::OrtKITensor *make_tensor(const std::vector<T> &value, const std::vector<int64_t> &shape) {
    auto *ptr = new T[value.size()];
    for (int i = 0; i < value.size(); ++i) {
        ptr[i] = value[i];
    }
    return new ortki::OrtKITensor((void *) ptr, ortki::TypeToDataType<T>(), shape);
}

ortki::OrtKITensorSeq *ortki_Split(ortki::OrtKITensor *input, ortki::OrtKITensor *split, int64_t axis) {
    ortki::OpExecutor Split("Split");
    Split.AddInput("input", input);
    Split.AddInput("split", split);
    Split.AddAttribute("axis", axis);

    Split.SetOutputSize(split->shape()[0]);
    auto results = Split.Run();
    return new ortki::OrtKITensorSeq(results);
}

ortki::OrtKITensor *ortki_ResizeWithSizes(ortki::OrtKITensor *X, ortki::OrtKITensor *roi, ortki::OrtKITensor *sizes,
                                          const char *coordinate_transformation_mode, float cubic_coeff_a,
                                          int64_t exclude_outside,
                                          float extrapolation_value, const char *mode, const char *nearest_mode) {
    ortki::OpExecutor Resize("Resize");
    Resize.AddInput("X", X);
    Resize.AddInput("roi", roi);
    Resize.AddOptionalInputEdge<float>();
    Resize.AddInput("sizes", sizes);
    Resize.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);
    Resize.AddAttribute("cubic_coeff_a", cubic_coeff_a);
    Resize.AddAttribute("exclude_outside", exclude_outside);
    Resize.AddAttribute("extrapolation_value", extrapolation_value);
    Resize.AddAttribute("mode", mode);
    Resize.AddAttribute("nearest_mode", nearest_mode);
    return new ortki::OrtKITensor(Resize.Run()[0]);
}


ortki::OrtKITensor *ortki_ResizeWithScales(ortki::OrtKITensor *X, ortki::OrtKITensor *roi, ortki::OrtKITensor *scales,
                                           const char *coordinate_transformation_mode, float cubic_coeff_a,
                                           int64_t exclude_outside,
                                           float extrapolation_value, const char *mode, const char *nearest_mode) {
    ortki::OpExecutor Resize("Resize");
    Resize.AddInput("X", X);
    Resize.AddInput("roi", roi);
    Resize.AddInput("scales", scales);
    Resize.AddOptionalInputEdge<float>();
    Resize.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);
    Resize.AddAttribute("cubic_coeff_a", cubic_coeff_a);
    Resize.AddAttribute("exclude_outside", exclude_outside);
    Resize.AddAttribute("extrapolation_value", extrapolation_value);
    Resize.AddAttribute("mode", mode);
    Resize.AddAttribute("nearest_mode", nearest_mode);
    return new ortki::OrtKITensor(Resize.Run()[0]);
}

ORTKI_API(ortki::OrtKITensor *)ortki_BatchNormalization(ortki::OrtKITensor *X, ortki::OrtKITensor *scale,
                                                        ortki::OrtKITensor *B, ortki::OrtKITensor *input_mean,
                                                        ortki::OrtKITensor *input_var, float epsilon, float momentum) {
    ortki::OpExecutor BatchNormalization("BatchNormalization");
    BatchNormalization.AddInput("X", X);
    BatchNormalization.AddInput("scale", scale);
    BatchNormalization.AddInput("B", B);
    BatchNormalization.AddInput("input_mean", input_mean);
    BatchNormalization.AddInput("input_var", input_var);
    BatchNormalization.AddAttribute("epsilon", epsilon);
    BatchNormalization.AddAttribute("momentum", momentum);
    BatchNormalization.SetOutputSize(1);
    return new ortki::OrtKITensor(BatchNormalization.Run()[0]);
}

ORTKI_API(ortki::OrtKITensor *)ortki_Upsample(ortki::OrtKITensor *X, ortki::OrtKITensor *scales, const char *mode) {
    ortki::OpExecutor Upsample("Upsample", 9);
    Upsample.AddInput("X", X);
    Upsample.AddInput("scales", scales);
    Upsample.AddAttribute("mode", mode);
    return new ortki::OrtKITensor(Upsample.Run()[0]);
}

ORTKI_API(ortki::OrtKITensorSeq *)ortki_LSTM(ortki::OrtKITensor *X, ortki::OrtKITensor *W, ortki::OrtKITensor *R,
                                             ortki::OrtKITensor *B, ortki::OrtKITensor *sequence_lens,
                                             ortki::OrtKITensor *initial_h, ortki::OrtKITensor *initial_c,
                                             ortki::OrtKITensor *P, float *activation_alpha, int activation_alpha_size,
                                             float *activation_beta, int activation_beta_size, const char **activations,
                                             int activations_size, float clip, const char *direction,
                                             int64_t hidden_size, int64_t input_forget, int64_t layout, bool has_clip, int64_t output_size) {
    ortki::OpExecutor LSTM("LSTM");
    LSTM.AddInput("X", X);
    LSTM.AddInput("W", W);
    LSTM.AddInput("R", R);
    LSTM.AddInput("B", B);
    LSTM.AddInput("sequence_lens", sequence_lens);
    LSTM.AddInput("initial_h", initial_h);
    LSTM.AddInput("initial_c", initial_c);
    LSTM.AddInput("P", P);
    if (activation_alpha_size > 0) {
        LSTM.AddAttribute("activation_alpha", ortki::ToVector(activation_alpha, activation_alpha_size));
    }
    if (activation_beta_size > 0) {
        LSTM.AddAttribute("activation_beta", ortki::ToVector(activation_beta, activation_beta_size));
    }
    LSTM.AddAttribute("activations", ortki::ToVector<const char *, std::string>(activations, activations_size));
    if (has_clip) {
        LSTM.AddAttribute("clip", clip);
    }
    LSTM.AddAttribute("direction", direction);
    LSTM.AddAttribute("hidden_size", hidden_size);
    LSTM.AddAttribute("input_forget", input_forget);
    LSTM.AddAttribute("layout", layout);
    LSTM.SetOutputSize(output_size);
    return new ortki::OrtKITensorSeq(LSTM.Run());
}