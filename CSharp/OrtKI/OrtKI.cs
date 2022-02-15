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

public static class OrtKI
{
    [DllImport("libortki.so")]
    private static extern IntPtr ortki_Binary(BinaryOp op, IntPtr a, IntPtr b);
    
    [DllImport("libortki.so")]
    private static extern IntPtr ortki_Unary(UnaryOp op, IntPtr input);

    [DllImport("libortki.so")]
    public static extern IntPtr ortki_Cast(IntPtr input, OrtDataType dataType);
    
    public static Tensor Binary(BinaryOp op, Tensor a, Tensor b)
    {
        var tensor = ortki_Binary(op, a.Handle, b.Handle);
        return new Tensor(tensor);
    }

    public static Tensor Unary(UnaryOp op, Tensor input)
    {
        var tensor = ortki_Unary(op, input.Handle);
        return new Tensor(tensor);
    }

    public static Tensor Cast(Tensor input, OrtDataType dataType)
    {
        return new Tensor(ortki_Cast(input.Handle, dataType));
    }
    
    public static Tensor Minimum(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.Min, a, b);
    }

    public static Tensor Maximum(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.Max, a, b);
    }

    public static Tensor Pow(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.Pow, a, b);
    }

    public static Tensor BitwiseAnd(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.BitwiseAnd, a, b);
    }

    public static Tensor BitwiseOr(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.BitwiseOr, a, b);
    }
    
    public static Tensor BitwiseXor(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.BitwiseXor, a, b);
    }
    
    public static Tensor LogicalAnd(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.LogicalAnd, a, b);
    }
    
    public static Tensor LogicalOr(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.LogicalOr, a, b);
    }
    
    public static Tensor LogicalXor(Tensor a, Tensor b)
    {
        return Binary(BinaryOp.LogicalXor, a, b);
    }

    public static Tensor Abs(Tensor input)
    {
        return Unary(UnaryOp.Abs, input);
    }
    
    public static Tensor Acos(Tensor input)
    {
        return Unary(UnaryOp.Acos, input);
    }

    public static Tensor Acosh(Tensor input)
    {
        return Unary(UnaryOp.Acosh, input);
    }

    public static Tensor Asin(Tensor input)
    {
        return Unary(UnaryOp.Asin, input);
    }

    public static Tensor Asinh(Tensor input)
    {
        return Unary(UnaryOp.Asinh, input);
    }

    public static Tensor Ceil(Tensor input)
    {
        return Unary(UnaryOp.Ceil, input);
    }

    public static Tensor Cos(Tensor input)
    {
        return Unary(UnaryOp.Cos, input);
    }

    public static Tensor Cosh(Tensor input)
    {
        return Unary(UnaryOp.Cosh, input);
    }

    public static Tensor Exp(Tensor input)
    {
        return Unary(UnaryOp.Exp, input);
    }

    public static Tensor Floor(Tensor input)
    {
        return Unary(UnaryOp.Floor, input);
    }

    public static Tensor Log(Tensor input)
    {
        return Unary(UnaryOp.Log, input);
    }

    public static Tensor Neg(Tensor input)
    {
        return Unary(UnaryOp.Neg, input);
    }

    public static Tensor Round(Tensor input)
    {
        return Unary(UnaryOp.Round, input);
    }

    public static Tensor Rsqrt(Tensor input)
    {
        return Unary(UnaryOp.Rsqrt, input);
    }

    public static Tensor Sin(Tensor input)
    {
        return Unary(UnaryOp.Sin, input);
    }

    public static Tensor Sinh(Tensor input)
    {
        return Unary(UnaryOp.Sinh, input);
    }

    public static Tensor Sign(Tensor input)
    {
        return Unary(UnaryOp.Sign, input);
    }

    public static Tensor Sqrt(Tensor input)
    {
        return Unary(UnaryOp.Sqrt, input);
    }

    public static Tensor Square(Tensor input)
    {
        return Unary(UnaryOp.Square, input);
    }
    
    public static Tensor Tanh(Tensor input)
    {
        return Unary(UnaryOp.Tanh, input);
    }
    
    public static Tensor BitwiseNot(Tensor input)
    {
        return Unary(UnaryOp.BitwiseNot, input);
    }
    
    public static Tensor LogicalNot(Tensor input)
    {
        return Unary(UnaryOp.LogicalNot, input);
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