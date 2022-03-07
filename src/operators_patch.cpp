#include <operators_patch.h>

ortki::OrtKITensorSeq *ortki_Split(ortki::OrtKITensor *input, ortki::OrtKITensor *split, long axis) {
    ortki::OpExecutor Split("Split");
    Split.AddInput("input", input);
    Split.AddInput("split", split);
    Split.AddAttribute("axis", axis);
    Split.SetOutputSize(split->shape()[0]);
    return new ortki::OrtKITensorSeq(Split.Run());
}

ortki::OrtKITensor * ortki_ResizeWithSizes(ortki::OrtKITensor *X, ortki::OrtKITensor *roi, ortki::OrtKITensor *sizes,
             const char *coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside,
             float extrapolation_value, const char *mode, const char *nearest_mode) {
    ortki::OpExecutor Resize("Resize");
    Resize.AddInput("X", X);
    Resize.AddInput("roi", roi);
    Resize.AddInput("sizes", sizes);
    Resize.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);
    Resize.AddAttribute("cubic_coeff_a", cubic_coeff_a);
    Resize.AddAttribute("exclude_outside", exclude_outside);
    Resize.AddAttribute("extrapolation_value", extrapolation_value);
    Resize.AddAttribute("mode", mode);
    Resize.AddAttribute("nearest_mode", nearest_mode);
    return new ortki::OrtKITensor(Resize.Run()[0]);
}


ortki::OrtKITensor * ortki_ResizeWithScales(ortki::OrtKITensor *X, ortki::OrtKITensor *roi, ortki::OrtKITensor *scales,
                                           const char *coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside,
                                           float extrapolation_value, const char *mode, const char *nearest_mode) {
    ortki::OpExecutor Resize("Resize");
    Resize.AddInput("X", X);
    Resize.AddInput("roi", roi);
    Resize.AddInput("scales", scales);
    Resize.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);
    Resize.AddAttribute("cubic_coeff_a", cubic_coeff_a);
    Resize.AddAttribute("exclude_outside", exclude_outside);
    Resize.AddAttribute("extrapolation_value", extrapolation_value);
    Resize.AddAttribute("mode", mode);
    Resize.AddAttribute("nearest_mode", nearest_mode);
    return new ortki::OrtKITensor(Resize.Run()[0]);
}

ORTKI_API(ortki::OrtKITensor *) ortki_BatchNormalization(ortki::OrtKITensor * X, ortki::OrtKITensor * scale, ortki::OrtKITensor * B, ortki::OrtKITensor * input_mean, ortki::OrtKITensor * input_var, float epsilon, float momentum)
{
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