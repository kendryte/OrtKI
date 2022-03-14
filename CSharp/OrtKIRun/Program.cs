// // See https://aka.ms/new-console-template for more information
//
// using OrtKISharp;
//
//

using OrtKISharp;
Console.WriteLine("Hello, World!");
var t = Tensor.FromScalar(1f);
var n = t + 1f;
Console.WriteLine("Hello, World!");

// var tensor1 = Tensor.MakeTensor(new[] {1, 2, 3}, new[] {3});
// Console.WriteLine(tensor1.ToArray<int>().ToString());
// Console.WriteLine(tensor1.ToArray<int>().Aggregate("", (s, i) => s + i + " "));
// var tensor2 = Tensor.MakeTensor(new[] {2, 2, 3}, new[] {3});
// var result = OrtKI.Binary(BinaryOp.Add, tensor1, tensor2);
// Console.WriteLine(result.ToArray<int>().Aggregate("", (s, i) => s + i + " "));
// var tensor3 = Tensor.MakeTensor(new[] {2.2, 2.2, 3.3}, new[] {3});
// var result_cast = OrtKI.Cast(tensor2, OrtDataType.Float);
// Console.WriteLine(result_cast.ToArray<float>().Aggregate("", (s, i) => s + i + " "));
// Console.WriteLine(result_cast.ToDense<int>().Aggregate("", (s, i) => s + i + " "));
