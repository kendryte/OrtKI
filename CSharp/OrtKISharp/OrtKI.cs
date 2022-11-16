using System.Reflection;
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

    public static Tensor Random(long[] shape, OrtDataType dt = OrtDataType.Float)
    {
        return RandomNormal((long)dt, 0.0f, 1.0f, new Random().NextSingle(), shape);
    }

    public static Tensor Random(params long[] shape)
    {
        return Random(shape, OrtDataType.Float);
    }

    public static Tensor Random(long n, OrtDataType dt = OrtDataType.Float)
    {
        return Random(new[] { n }, dt);
    }

#if DEBUG
    static string nativeRid =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "win-x64" :
        RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "linux-x64" :
        RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ?
            RuntimeInformation.ProcessArchitecture == Architecture.X64
                ? "osx-x64" : "osx-arm64" :
        "any";

    static string nativeLib =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "ortki.dll" :
        RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "libortki.dylib" :
        "libortki.so";

    private static bool _loadDllSet = false;

    public static void LoadDLL()
    {
        if (_loadDllSet) return;
        _loadDllSet = true;

        var assembly = typeof(Tensor).Assembly;
        NativeLibrary.SetDllImportResolver(assembly, (string libraryName, Assembly assembly, DllImportSearchPath? searchPath) =>
        {
            if (libraryName == Native.LibraryName)
            {
                var assemblyFolder = Path.GetDirectoryName(assembly.Location);
                var dll = Path.Join(assemblyFolder, "../../../../../out/install/x64-Debug/bin/", nativeLib);
                return NativeLibrary.Load(dll, assembly, null);
            }
            else
            {
                return NativeLibrary.Load(libraryName, assembly, searchPath);
            }
        });
    }
#endif
}
