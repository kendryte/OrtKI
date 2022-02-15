using System.Runtime.InteropServices;

namespace OrtKISharp;

public enum OrtDataType : byte
{
    Undefined = 0,
    Float = 1,
    UInt8 = 2,
    Int8 = 3,
    UInt16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    UInt32 = 12,
    UInt64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    BFloat16 = 16
}

public static class TypeUtil
{
    private static readonly Dictionary<OrtDataType, int> _DataTypeToLengths = new()
    {
        { OrtDataType.Bool, 1 },
        { OrtDataType.UInt8, 1 },
        { OrtDataType.UInt16, 2 },
        { OrtDataType.UInt32, 4 },
        { OrtDataType.UInt64, 8 },
        { OrtDataType.Int8, 1 },
        { OrtDataType.Int16, 2 },
        { OrtDataType.Int32, 4 },
        { OrtDataType.Int64, 8 },
        { OrtDataType.Float16, 2 },
        { OrtDataType.BFloat16, 2 },
        { OrtDataType.Float, 4 },
        { OrtDataType.Double, 8 },
    };

    private static readonly Dictionary<RuntimeTypeHandle, OrtDataType> _typeToDataTypes = new()
    {
        { typeof(bool).TypeHandle, OrtDataType.Bool },
        { typeof(sbyte).TypeHandle, OrtDataType.Int8 },
        { typeof(byte).TypeHandle, OrtDataType.UInt8 },
        { typeof(int).TypeHandle, OrtDataType.Int32 },
        { typeof(uint).TypeHandle, OrtDataType.UInt32 },
        { typeof(long).TypeHandle, OrtDataType.Int64 },
        { typeof(ulong).TypeHandle, OrtDataType.UInt64 },
        { typeof(float).TypeHandle, OrtDataType.Float },
        { typeof(double).TypeHandle, OrtDataType.Double },
        { typeof(char).TypeHandle, OrtDataType.String },
        // todo:bf16 and float 16
    };
    
    private static readonly Dictionary<OrtDataType, Type> _dataTypesToType = new()
    {
        { OrtDataType.Bool, typeof(bool) },
        { OrtDataType.Int8, typeof(sbyte) },
        { OrtDataType.UInt8, typeof(byte) },
        { OrtDataType.Int32, typeof(int) },
        { OrtDataType.UInt32, typeof(uint) },
        { OrtDataType.Int64, typeof(long) },
        { OrtDataType.UInt64, typeof(ulong) },
        { OrtDataType.Float, typeof(float) },
        { OrtDataType.Double, typeof(double) },
    };

    public static int GetLength(OrtDataType dt)
    {
        if (_DataTypeToLengths.TryGetValue(dt, out var type))
        {
            return type;
        }
        throw new ArgumentOutOfRangeException("Unsupported OrtDataType: " + dt);
    }
    
    public static OrtDataType FromType(Type t)
    {
        if (_typeToDataTypes.TryGetValue(t.TypeHandle, out var type))
        {
            return type;
        }
        throw new ArgumentOutOfRangeException("Unsupported OrtDataType: " + t);
    }

    public static Type ToType(OrtDataType t)
    {
        if (_dataTypesToType.TryGetValue(t, out var type))
        {
            return type;
        }
        throw new ArgumentOutOfRangeException("Unsupported OrtDataType: " + t);
    }
    
    public static byte[] GetBytes<T>(ReadOnlySpan<T> span)
        where T : unmanaged
        => MemoryMarshal.AsBytes(span).ToArray();
}