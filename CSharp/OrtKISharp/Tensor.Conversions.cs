namespace OrtKISharp;

public partial class Tensor
{
    public static implicit operator Tensor(sbyte x) => FromScalar(x);
    
    public static implicit operator Tensor(short x) => FromScalar(x);
    
    public static implicit operator Tensor(int x) => FromScalar(x);
    
    public static implicit operator Tensor(long x) => FromScalar(x);
    
    public static implicit operator Tensor(byte x) => FromScalar(x);
    
    public static implicit operator Tensor(ushort x) => FromScalar(x);
    
    public static implicit operator Tensor(uint x) => FromScalar(x);
    
    public static implicit operator Tensor(ulong x) => FromScalar(x);
    
    public static implicit operator Tensor(float x) => FromScalar(x);
    
    public static implicit operator Tensor(int[] x) => MakeTensor(x);
    
    public static implicit operator Tensor(long[] x) => MakeTensor(x);
    
    public static implicit operator Tensor(float[] x) => MakeTensor(x);
    
}