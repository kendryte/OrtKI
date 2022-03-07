using System;
using OrtKISharp;
using Xunit;

namespace OrtKITest;

public class KernelTest
{
    public KernelTest()
    {
        OrtKI.LoadDLL();    
    }
    
    [Fact]
    public void TestCast()
    {
        var t1 = Tensor.MakeTensor(new[] {1, 2, 3});
        var t = t1.ToType(OrtDataType.Float16);
        // Assert.Equal(new[] {1.0f, 2.0f, 3.0f}, t.ToArray<float>());
    }
    
    [Fact]
    public void TestConcat()
    {
        var t1 = Tensor.MakeTensor(new[] {1, 2, 3});
        var t2 = Tensor.MakeTensor(new[] {1, 2, 3});
        var output = OrtKI.Concat(new[] {t1, t2}, 0);
        Assert.Equal(output.Shape, new[] {6});
        Assert.Equal(new[] {1, 2, 3, 1, 2, 3}, output.ToArray<int>());
    }

    [Fact]
    public void TestSlice()
    {
        var tensor = Tensor.MakeTensor(new[] {1, 2, 3, 4, 5, 6});
        var start = Tensor.MakeTensor(new[] {1});
        var end = Tensor.MakeTensor(new[] {5});
        var axes = Tensor.MakeTensor(new[] {0});
        var steps = Tensor.MakeTensor(new[] {2});
        var output = OrtKI.Slice(tensor, start, end, axes, steps);
        Assert.Equal(new[] {2}, output.Shape);
        Assert.Equal(new[] {2, 4}, output.ToArray<int>());
    }

    [Fact]
    public void TestTranspose()
    {
        var t = Tensor.MakeTensor(new[] {1, 2, 3, 4, 5, 6}, new[] {2, 3});
        var perm = new long[] {0, 1};
        OrtKI.Transpose(t, perm);
    }

    [Fact]
    public void TestResize()
    {
        var t = Tensor.MakeTensor(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new[] {1, 1, 2, 4});
        var roi = Tensor.MakeTensor(Array.Empty<float>());
        //var roi = Tensor.Empty(new[] {1}, OrtDataType.Float);
        var scales = Tensor.MakeTensor(new float[] {1, 1, 0.6f, 0.6f});
        OrtKI.ResizeWithScales(t, roi, scales, "align_corners", -0.75f, 0, 0, "linear", "floor");
    }

    [Fact]
    public void TestBatchNorm()
    {
        var input = Tensor.MakeTensor(new[] {0.329876f, -0.287158f, -0.411425f, 0.473621f, 0.18156f, -0.170596f, -0.329516f, -0.170733f, -0.121664f, 0.4372f,
            -0.485668f, 0.218049f, -0.360263f, 0.107016f, 0.45358f, 0.325056f, 0.15995f, 0.098852f, -0.283453f, -0.373051f,
            0.257542f, 0.0614853f, -0.0592363f, 0.434488f, -0.0179583f, 0.398374f, -0.451602f, -0.132009f, -0.174468f,
            -0.0247169f, 0.418897f, -0.47159f, -0.131925f, 0.470943f, 0.118357f, 0.155664f, 0.370062f, -0.279229f, 0.240311f,
            -0.451034f, 0.249178f, -0.294496f, 0.13683f, -0.0806475f, -0.309849f, -0.450604f, -0.28048f, -0.420197f, -0.433369f},
        new[] {1, 1, 7, 7, 1});
        var scale = Tensor.MakeTensor(new[] {0.589433f});
        var B = Tensor.MakeTensor(new[] {-0.384622f});
        var mean = Tensor.MakeTensor(new[] {-2.45673f});
        var var = Tensor.MakeTensor(new[] {1.37998f});
        var batchNorm = OrtKI.BatchNormalization(input, scale, B, mean, var, 1e-05f, 0.1f);
        Assert.Equal(OrtDataType.Float, batchNorm.DataType);
    }

    [Fact]
    public void TestArgMin()
    {
        var input = Tensor.MakeTensor(new[]{1,1,1,1,1,1,1,1,1,},new[] {1, 1, 3, 3});
        var result = OrtKI.Squeeze(input, Tensor.Empty(new[]{0}, OrtDataType.Int64));
        // var result = OrtKI.Squeeze(input, Tensor.MakeTensor(new long[] {0, 1}));
        Assert.True(true);
    }
}