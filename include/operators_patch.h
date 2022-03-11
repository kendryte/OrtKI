#include "op_executor.h"
#include "tensor.h"

// ops which be Not suitable for auto-generated
ORTKI_API(ortki::OrtKITensorSeq) * ortki_Split(ortki::OrtKITensor * input, ortki::OrtKITensor * split, int64_t axis);

// only one of size and scale can be passed
ORTKI_API(ortki::OrtKITensor *) ortki_ResizeWithSizes(ortki::OrtKITensor * X, ortki::OrtKITensor * roi, ortki::OrtKITensor * sizes, const char* coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, const char* mode, const char* nearest_mode);

ORTKI_API(ortki::OrtKITensor *) ortki_ResizeWithScales(ortki::OrtKITensor * X, ortki::OrtKITensor * roi, ortki::OrtKITensor * scales, const char* coordinate_transformation_mode, float cubic_coeff_a, int64_t exclude_outside, float extrapolation_value, const char* mode, const char* nearest_mode);

// training mode set false and must spec only one output
ORTKI_API(ortki::OrtKITensor *) ortki_BatchNormalization(ortki::OrtKITensor * X, ortki::OrtKITensor * scale, ortki::OrtKITensor * B, ortki::OrtKITensor * input_mean, ortki::OrtKITensor * input_var, float epsilon, float momentum);