using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OrtKISharp;

public partial class Tensor
{
    public static Tensor FromScalar<T>(T x)
        where T : unmanaged
    {
        return MakeTensor(new[] { x }, Array.Empty<long>());
    }

    public static Tensor MakeTensor<T>(T[] array, long[] shape)
        where T : unmanaged
    {
        return MakeTensor(array.AsMemory(), shape);
    }

    public static Tensor MakeTensor<T>(T[] array)
        where T : unmanaged
    {
        return MakeTensor<T>(array.AsMemory(), new[] { (long)array.Length });
    }

    public static Tensor MakeTensor<T>(Memory<T> memory, long[] shape)
        where T : unmanaged
    {
        return MakeTensor(memory, TypeUtil.FromType(typeof(T)), shape);
    }

    public static Tensor MakeTensor<T>(Memory<T> memory, OrtDataType dataType, long[] shape)
        where T : unmanaged
    {
        return MakeTensor(memory.Pin(), dataType, shape);
    }

    public static unsafe Tensor MakeTensor(MemoryHandle memoryHandle, OrtDataType dataType, long[] shape)
    {
        fixed (long* shapePtr = shape)
        {
            var tensor = Native.make_tensor(memoryHandle.Pointer, dataType, shapePtr, (nuint)shape.Length);
            tensor._memoryHandle = memoryHandle;
            return tensor;
        }
    }

    public static unsafe Tensor Empty(long[] shape, OrtDataType dataType = OrtDataType.Float)
    {
        fixed (long* shape_ptr = shape)
        {
            return Native.make_tensor_empty(dataType, shape_ptr, (nuint)shape.Length);
        }
    }
}
