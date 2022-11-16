using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OrtKISharp;

internal partial class Native
{
    [DllImport(LibraryName)]
    public static extern TensorSeq ortki_Split(Tensor input, Tensor split, long axis);

    [DllImport(LibraryName)]
    public static extern Tensor ortki_ResizeWithScales(Tensor X, Tensor roi, Tensor scales, string coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, string mode, string nearest_mode);

    [DllImport(LibraryName)]
    public static extern Tensor ortki_ResizeWithSizes(Tensor X, Tensor roi, Tensor sizes, string coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, string mode, string nearest_mode);

    [DllImport(LibraryName)]
    public static extern Tensor ortki_BatchNormalization(Tensor X, Tensor scale, Tensor B, Tensor input_mean, Tensor input_var, float epsilon, float momentum);

    [DllImport(LibraryName)]
    public static extern TensorSeq ortki_LSTM(Tensor X, Tensor W, Tensor R, Tensor B, Tensor sequence_lens, Tensor initial_h, Tensor initial_c, Tensor P, float[] activation_alpha, nuint activation_alpha_size, float[] activation_beta, nuint activation_beta_size, string[] activations, nuint activations_size, float clip, string direction, long hidden_size, long input_forget, long layout, bool has_clip, long output_size);
}
