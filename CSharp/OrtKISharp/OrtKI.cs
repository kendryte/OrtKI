using System.Runtime.InteropServices;

namespace OrtKISharp;

public enum BinaryOp
{
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Min,
    Max,
    Pow,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
}

public enum UnaryOp
{
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Floor,
    Log,
    Neg,
    Round,
    Rsqrt,
    Sin,
    Sinh,
    Sign,
    Sqrt,
    Square,
    Tanh,
    BitwiseNot,
    LogicalNot,
}

public partial class OrtKI
{
    // [DllImport("ortki")]
    // private static extern IntPtr ortki_Binary(BinaryOp op, IntPtr a, IntPtr b);
    //
    // [DllImport("ortki")]
    // private static extern IntPtr ortki_Unary(UnaryOp op, IntPtr input);
    //
    // public static Tensor Binary(BinaryOp op, Tensor a, Tensor b)
    // {
    //     var tensor = ortki_Binary(op, a.Handle, b.Handle);
    //     return new Tensor(tensor);
    // }
    //
    // public static Tensor Unary(UnaryOp op, Tensor input)
    // {
    //     var tensor = ortki_Unary(op, input.Handle);
    //     return new Tensor(tensor);
    // }
    //
    [DllImport("ortki")]
    private static extern IntPtr ortki_Split(IntPtr input, IntPtr split, long axis);
    
    public static Tensor[] Split(Tensor input, Tensor split, long axis){
        var _tensor = ortki_Split(input.Handle, split.Handle, axis);
        return new TensorSeq(_tensor).ToTensorArray();
    }

    public static Tensor Square(Tensor input)
    {
        return Mul(input, input);
    }

    public static Tensor Rsqrt(Tensor input)
    {
        return Reciprocal(Sqrt(input));
    }

    public static Tensor LeftShift(Tensor a, Tensor b)
    {
        return BitShift(a, b, "LEFT");
    }
    
    public static Tensor RightShift(Tensor a, Tensor b)
    {
        return BitShift(a, b, "RIGHT");
    }

    public static Tensor Random(int[] shape, OrtDataType dt = OrtDataType.Float)
    {
        return RandomNormal((long)dt, 0.0f, 1.0f, new Random().NextSingle(), 
            shape.Select(x => (long)x).ToArray());
    }
    
    public static Tensor Random(params int[] shape)
    {
        return Random(shape, OrtDataType.Float);
    }

    public static Tensor Random(int n, OrtDataType dt = OrtDataType.Float)
    {
        return Random(new[]{n}, dt);
    }
    
    
    [DllImport("ortki")]
    private static extern IntPtr ortki_ResizeWithScales(IntPtr X, IntPtr roi, IntPtr scales, String coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, String mode, String nearest_mode);
    
    [DllImport("ortki")]
    private static extern IntPtr ortki_ResizeWithSizes(IntPtr X, IntPtr roi, IntPtr sizes, String coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, String mode, String nearest_mode);
    
    public static Tensor ResizeWithScales(Tensor X, Tensor roi, Tensor scales, String coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, String mode, String nearest_mode){
        var _tensor = ortki_ResizeWithScales(X.Handle, roi.Handle, scales.Handle, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
        return new Tensor(_tensor);
    }
    
    public static Tensor ResizeWithSizes(Tensor X, Tensor roi, Tensor sizes, String coordinate_transformation_mode, float cubic_coeff_a, long exclude_outside, float extrapolation_value, String mode, String nearest_mode){
        var _tensor = ortki_ResizeWithSizes(X.Handle, roi.Handle, sizes.Handle, coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, mode, nearest_mode);
        return new Tensor(_tensor);
    }
    
    [DllImport("ortki")]
    private static extern IntPtr ortki_BatchNormalization(IntPtr X, IntPtr scale, IntPtr B, IntPtr input_mean, IntPtr input_var, float epsilon, float momentum);

    public static Tensor BatchNormalization(Tensor X, Tensor scale, Tensor B, Tensor input_mean, Tensor input_var, float epsilon, float momentum){
        var _tensor = ortki_BatchNormalization(X.Handle, scale.Handle, B.Handle, input_mean.Handle, input_var.Handle, epsilon, momentum);
        return new Tensor(_tensor);
    }
    
    [DllImport("ortki")]
    private static extern IntPtr ortki_LSTM(IntPtr X, IntPtr W, IntPtr R, IntPtr B, IntPtr sequence_lens, IntPtr initial_h, IntPtr initial_c, IntPtr P, float[] activation_alpha, int activation_alpha_size, float[] activation_beta, int activation_beta_size, String[] activations, int activations_size, float clip, String direction, long hidden_size, long input_forget, long layout, bool has_clip, long output_size);
    
    public static Tensor[] LSTM(Tensor X, Tensor W, Tensor R, Tensor B, Tensor sequence_lens, Tensor initial_h, Tensor initial_c, Tensor P, float[] activation_alpha, float[] activation_beta, String[] activations, float clip, String direction, long hidden_size, long input_forget, long layout, bool has_clip, long output_size)
    {
        var _tensor = ortki_LSTM(X.Handle, W.Handle, R.Handle, B.Handle, sequence_lens.Handle, initial_h.Handle, initial_c.Handle, P.Handle, activation_alpha, activation_alpha.Length, activation_beta, activation_beta.Length, activations, activations.Length, clip, direction, hidden_size, input_forget, layout, has_clip, output_size);
        return new TensorSeq(_tensor).ToTensorArray();
    }
    
    public static void LoadDLL()
    {
        var assembly = typeof(Tensor).Assembly;
        var path = new DllImportSearchPath();
        var ok = NativeLibrary.TryLoad("/home/homura/Code/OrtKI/cmake-build-debug/libortki.so", assembly, null, out var res1);
        if (!ok)
        {
            throw new DllNotFoundException("libortki not found");
        }
    }
}