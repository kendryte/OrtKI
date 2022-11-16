using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OrtKISharp;

public partial class OrtKI
{
    public static Tensor[] Split(Tensor input, Tensor split, long axis)
    {
        var _tensor = Native.ortki_Split(input, split, axis);
        return _tensor.ToTensorArray();
    }

    public static Tensor ResizeWithScales(Tensor X, Tensor roi, Tensor scales, string coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, string mode, string nearest_mode)
    {
        var _tensor = Native.ortki_ResizeWithScales(X, roi, scales, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
        return _tensor;
    }

    public static Tensor ResizeWithSizes(Tensor X, Tensor roi, Tensor sizes, string coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, string mode, string nearest_mode)
    {
        var _tensor = Native.ortki_ResizeWithSizes(X, roi, sizes, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
        return _tensor;
    }

    public static Tensor BatchNormalization(Tensor X, Tensor scale, Tensor B, Tensor input_mean, Tensor input_var, float epsilon, float momentum)
    {
        var _tensor = Native.ortki_BatchNormalization(X, scale, B, input_mean, input_var, epsilon, momentum);
        return _tensor;
    }

    public static Tensor[] LSTM(Tensor X, Tensor W, Tensor R, Tensor B, Tensor sequence_lens, Tensor initial_h, Tensor initial_c, Tensor P, float[] activation_alpha, float[] activation_beta, string[] activations, float clip, string direction, long hidden_size, long input_forget, long layout, bool has_clip, long output_size)
    {
        var _tensor = Native.ortki_LSTM(X, W, R, B, sequence_lens, initial_h, initial_c, P, activation_alpha, (nuint)activation_alpha.Length, activation_beta, (nuint)activation_beta.Length, activations, (nuint)activations.Length, clip, direction, hidden_size, input_forget, layout, has_clip, output_size);
        return _tensor.ToTensorArray();
    }
}
