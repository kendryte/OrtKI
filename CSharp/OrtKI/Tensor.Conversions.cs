namespace OrtKISharp;

public partial class Tensor
{
    public static implicit operator Tensor(sbyte x) => new(x);
    
    public static implicit operator Tensor(short x) => new(x);
    
    public static implicit operator Tensor(int x) => new(x);
    
    public static implicit operator Tensor(long x) => new(x);
    
    public static implicit operator Tensor(byte x) => new(x);
    
    public static implicit operator Tensor(ushort x) => new(x);
    
    public static implicit operator Tensor(uint x) => new(x);
    
    public static implicit operator Tensor(ulong x) => new(x);
    
    public static implicit operator Tensor(float x) => new(x);
    
}