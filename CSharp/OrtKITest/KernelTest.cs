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
}