using System.Runtime.InteropServices;

namespace OrtKI;

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
    LeftShift,
    RightShift,
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
    [DllImport("ortki")]
    extern static private IntPtr ortki_Binary(BinaryOp op, IntPtr a, IntPtr b);
    
    [DllImport("ortki")]
    extern static private IntPtr ortki_Unary(UnaryOp op, IntPtr input);

    static public Tensor Binary(BinaryOp op, Tensor a, Tensor b)
    {
        var tensor = ortki_Binary(op, a.Handle, b.Handle);
        return new Tensor(tensor);
    }

    static public Tensor Unary(UnaryOp op, Tensor input)
    {
        var tensor = ortki_Unary(op, input.Handle);
        return new Tensor(tensor);
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