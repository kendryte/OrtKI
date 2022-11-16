namespace OrtKISharp;

public partial class Tensor
{
    public static Tensor operator +(Tensor lhs, Tensor rhs) => OrtKI.Add(lhs, rhs);

    public static Tensor operator -(Tensor lhs, Tensor rhs) => OrtKI.Sub(lhs, rhs);

    public static Tensor operator *(Tensor lhs, Tensor rhs) => OrtKI.Mul(lhs, rhs);

    public static Tensor operator /(Tensor lhs, Tensor rhs) => OrtKI.Div(lhs, rhs);

    public static Tensor operator %(Tensor lhs, Tensor rhs) => OrtKI.Mod(lhs, rhs, 0);

    public static Tensor operator -(Tensor x) => OrtKI.Neg(x);
}
