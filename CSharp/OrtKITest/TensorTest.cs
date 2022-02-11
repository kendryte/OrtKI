using System.Linq;
using Microsoft.Extensions.Hosting;
using OrtKI;
using Xunit;

namespace OrtKITest;

public class TensorTest
{

    public TensorTest(IHost host)
    {
        OrtKI.OrtKI.LoadDLL();
    }
    
    [Fact]
    public void TestAttribute()
    {
        var tensor = Tensor.MakeTensor(new[] {1, 3, 4, 2, 5, 6}, new[] {2, 3});
        Assert.Equal(OrtDataType.Int32, tensor.DataType);
        Assert.Equal(2, tensor.Rank);
        Assert.Equal(new[] {2, 3}, tensor.Shape);
        Assert.Equal(new[] {1, 3, 4, 2, 5, 6}, tensor.ToDense<int>().ToArray());
    }

    [Fact]
    public void TestForCompute()
    {
        var tensor1 = Tensor.MakeTensor(new[] {1, 2, 3}, new[] {3});
        var tensor2 = Tensor.MakeTensor(new[] {2, 2, 3}, new[] {3});
        var result = OrtKI.OrtKI.Binary(BinaryOp.Add, tensor1, tensor2);
        Assert.Equal(new[] {3, 4, 6}, result.ToDense<int>().ToArray());
    }
}