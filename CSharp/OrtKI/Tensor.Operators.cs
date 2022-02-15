namespace OrtKISharp;

public partial class Tensor
{
    public static Tensor operator +(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.Add, lhs, rhs);

    public static Tensor operator -(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.Sub, lhs, rhs);

    public static Tensor operator *(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.Mul, lhs, rhs);

    public static Tensor operator /(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.Div, lhs, rhs);

    public static Tensor operator %(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.Mod, lhs, rhs);

    public static Tensor operator &(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.BitwiseAnd, lhs, rhs);

    public static Tensor operator |(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.BitwiseOr, lhs, rhs);

    public static Tensor operator ^(Tensor lhs, Tensor rhs) => OrtKI.Binary(BinaryOp.BitwiseXor, lhs, rhs);
}