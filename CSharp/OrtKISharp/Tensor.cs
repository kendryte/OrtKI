using System.Collections;
using System.Reflection.Metadata;
using System.Runtime.InteropServices;
using System.Linq;
using System.Reflection;
using System.Buffers;

namespace OrtKISharp;

public sealed partial class Tensor : SafeHandle, IEquatable<Tensor>
{
    internal MemoryHandle? _memoryHandle;

    private long[]? _shape;

    internal Tensor()
        : this(IntPtr.Zero)
    {
    }

    internal Tensor(IntPtr handle)
        : base(handle, true)
    {
    }

    public override bool IsInvalid => handle == IntPtr.Zero;

    public OrtDataType DataType => Native.tensor_data_type(this);

    public int Rank => (int)Native.tensor_rank(this);

    public long Length => (long)Native.tensor_length(this);

    public unsafe long[] Shape
    {
        get
        {
            if (_shape == null)
            {
                var rank = Native.tensor_rank(this);
                var shape = new long[rank];
                fixed (long* shapePtr = shape)
                {
                    Native.tensor_shape(this, shapePtr);
                }

                _shape = shape;
                return shape;
            }

            return _shape;
        }
    }

    public unsafe Span<byte> BytesBuffer => GetBuffer<byte>();

    public unsafe Span<T> GetBuffer<T>() where T : unmanaged
    {
        void* ptr = Native.tensor_buffer(this, out nuint length);
        return new Span<T>(ptr, (int)(length / (nuint)sizeof(T)));
    }

    public unsafe T[] ToArray<T>() where T : unmanaged
    {
        return MemoryMarshal.Cast<byte, T>(BytesBuffer).ToArray();
    }

    public Tensor Cast(OrtDataType dataType)
    {
        if (DataType == dataType)
        {
            return this;
        }

        return Native.tensor_to_type(this, dataType);
    }

    public void Reshape(long[] shape)
    {
        Native.tensor_reshape(this, shape, (nuint)shape.Length);
        _shape = shape;
    }

    public Tensor BroadcastTo(long[] shape)
    {
        return this + Empty(shape);
    }

    public bool Equals(Tensor? other)
    {
        if (ReferenceEquals(null, other)) return false;
        if (ReferenceEquals(this, other)) return true;
        if (!Shape.SequenceEqual(other.Shape) || DataType != other.DataType) return false;
        return BytesBuffer.SequenceEqual(other.BytesBuffer);
    }

    public override bool Equals(object? obj)
    {
        if (obj is Tensor other)
        {
            return Equals(this, other);
        }

        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(handle);
    }

    protected override bool ReleaseHandle()
    {
        Native.tensor_dispose(handle);
        _memoryHandle?.Dispose();
        _memoryHandle = null;
        return true;
    }
}

internal sealed class TensorSeq : SafeHandle
{
    internal TensorSeq()
        : this(IntPtr.Zero)
    {
    }

    internal TensorSeq(IntPtr handle)
        : base(handle, true)
    {
    }

    public Tensor this[int index] => Native.tensor_seq_get_value(this, (nuint)index);

    public override bool IsInvalid => handle == IntPtr.Zero;

    public Tensor[] ToTensorArray()
    {
        var array = new Tensor[Native.tensor_seq_size(this)];
        for (int i = 0; i < array.Length; i++)
        {
            array[i] = this[i];
        }

        return array;
    }

    protected override bool ReleaseHandle()
    {
        Native.tensor_seq_dispose(handle);
        return true;
    }
}
