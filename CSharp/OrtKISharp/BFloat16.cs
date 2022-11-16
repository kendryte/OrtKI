using System.Runtime.InteropServices;

namespace OrtKISharp;

public class BFloat16 : IDisposable, IEquatable<BFloat16>
{
    [DllImport("ortki")]
    internal static extern unsafe IntPtr make_bf16(float v);
    
    [DllImport("ortki")]
    internal static extern unsafe float bf16_to_float(IntPtr h);
    
    [DllImport("ortki")]
    internal static extern unsafe void bf16_dispose(IntPtr h);

    [DllImport("ortki")]
    internal static extern unsafe ushort bf16_to_uint16(IntPtr h);

    private IntPtr Handle;

    public BFloat16(float v)
    {
        Handle = make_bf16(v);
    }
    
    public float ToFloat()
    {
        return bf16_to_float(Handle);
    }

    public ushort ToUInt16()
    {
        return bf16_to_uint16(Handle);
    }
    
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    
    protected virtual void Dispose(bool disposing)
    {
        if (Handle != IntPtr.Zero)
        {
            bf16_dispose(Handle);
            Handle = IntPtr.Zero;
        }
    }

    public bool Equals(BFloat16? other)
    {
        return ToUInt16() == other.ToUInt16();
    }
}