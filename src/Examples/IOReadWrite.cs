using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using TorchSharp;

using static TorchSharp.torch;

using SkiaSharp;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        internal static void Main(string[] args)
        {
            var filename = args[0];

            Console.WriteLine($"Reading file {filename}");

            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

            var img = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB);

            Console.WriteLine($"Image has {img.shape[0]} colour channels with dimensions {img.shape[1]}x{img.shape[2]}");

            var transformed = torchvision.transforms.Compose(
                torchvision.transforms.HorizontalFlip(),
                torchvision.transforms.Crop(0, 0, 816, 816),
                torchvision.transforms.Rotate(20) ,
                torchvision.transforms.Grayscale()
                ).forward(img);

            torchvision.io.write_image(transformed, "image_transformed.jpg", torchvision.ImageFormat.Jpeg);
        }
    }
}
