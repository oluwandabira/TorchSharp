using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;

using TorchSharp;
using static TorchSharp.torch;
using System.Runtime.InteropServices;

using SkiaSharp;
using System.Diagnostics;

namespace TorchSharp.Examples
{
    class IOReadWrite
    {
        class SkiaImager : torchvision.io.IImager
        {
            public torchvision.io.ImageFormat DetectFormat(byte[] bytes)
            {
                throw new NotImplementedException();
            }
            public Tensor DecodeImage(byte[] image, torchvision.io.ImageFormat format,  torchvision.io.ImageReadMode mode)
            {
                using (var stream = new SKManagedStream(new MemoryStream(image)))
                using (var bitmap = SKBitmap.Decode(stream)) {
                    using (var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap))) {

                        var channels = 3; // TODO: Support grayscale, too.

                        return inputTensor.reshape(channels, bitmap.Height, bitmap.Width);
                    }
                }
            }
            public Tensor DecodeImage(byte[] image, torchvision.io.ImageReadMode mode)
            {
                using (var stream = new SKManagedStream(new MemoryStream(image)))
                using (var bitmap = SKBitmap.Decode(stream)) {
                    using (var inputTensor = torch.tensor(GetBytesWithoutAlpha(bitmap))) {

                        var channels = 3; // TODO: Support grayscale, too.

                        return inputTensor.reshape(channels, bitmap.Height, bitmap.Width);
                    }
                }
            }

            private static byte[] GetBytesWithoutAlpha(SKBitmap bitmap)
            {
                var height = bitmap.Height;
                var width = bitmap.Width;

                var inputBytes = bitmap.Bytes;

                if (bitmap.ColorType == SKColorType.Gray8)
                    return inputBytes;

                if (bitmap.BytesPerPixel != 4 && bitmap.BytesPerPixel != 1)
                    throw new ArgumentException("Conversion only supports grayscale and ARGB");

                var channelLength = height * width;

                var channelCount = 3;

                int inputBlue = 0, inputGreen = 0, inputRed = 0;
                int outputRed = 0, outputGreen = channelLength, outputBlue = channelLength * 2;

                switch (bitmap.ColorType) {
                case SKColorType.Bgra8888:
                    inputBlue = 0;
                    inputGreen = 1;
                    inputRed = 2;
                    break;

                default:
                    throw new NotImplementedException($"Conversion from {bitmap.ColorType} to bytes");
                }
                var outBytes = new byte[channelCount * channelLength];

                for (int i = 0, j = 0; i < channelLength; i += 1, j += 4) {
                    outBytes[outputRed + i] = inputBytes[inputRed + j];
                    outBytes[outputGreen + i] = inputBytes[inputGreen + j];
                    outBytes[outputBlue + i] = inputBytes[inputBlue + j];
                }

                return outBytes;
            }

            private static SKBitmap GetBitmapFromBytes(byte[] inputBytes, int height, int width, SKColorType colorType)
            {
                var result = new SKBitmap();

                var channelLength = height * width;

                var channelCount = 0;

                int inputRed = 0, inputGreen = channelLength, inputBlue = channelLength * 2;
                int outputBlue = 0, outputGreen = 0, outputRed = 0, outputAlpha = 0;

                switch (colorType) {
                case SKColorType.Bgra8888:
                    outputBlue = 0;
                    outputGreen = 1;
                    outputRed = 2;
                    outputAlpha = 3;
                    channelCount = 3;
                    break;

                case SKColorType.Gray8:
                    channelCount = 1;
                    break;

                default:
                    throw new NotImplementedException($"Conversion from {colorType} to bytes");
                }

                byte[] outBytes = null;

                if (channelCount == 1) {

                    // Greyscale

                    outBytes = inputBytes;
                } else {

                    outBytes = new byte[(channelCount + 1) * channelLength];

                    for (int i = 0, j = 0; i < channelLength; i += 1, j += 4) {
                        outBytes[outputRed + j] = inputBytes[inputRed + i];
                        outBytes[outputGreen + j] = inputBytes[inputGreen + i];
                        outBytes[outputBlue + j] = inputBytes[inputBlue + i];
                        outBytes[outputAlpha + j] = 255;
                    }
                }

                // pin the managed array so that the GC doesn't move it
                var gcHandle = GCHandle.Alloc(outBytes, GCHandleType.Pinned);

                // install the pixels with the color type of the pixel data
                var info = new SKImageInfo((int)width, (int)height, colorType, SKAlphaType.Unpremul);
                result.InstallPixels(info, gcHandle.AddrOfPinnedObject(), info.RowBytes, delegate { gcHandle.Free(); }, null);

                return result;
            }

            public byte[] EncodeImage(Tensor image, torchvision.io.ImageFormat format)
            {
                var channels = image.shape[0];

                var data = image.data<byte>().ToArray();
                var bitmap = GetBitmapFromBytes(data, (int)image.shape[1], (int)image.shape[2], channels == 1 ? SKColorType.Gray8 : SKColorType.Bgra8888);
                using (var stream = new MemoryStream()) {
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                    stream.Flush();
                    return stream.ToArray();
                }
            }
        }

        internal static void Main(string[] args)
        {
            var filename = @"c:\Users\niklasg\Downloads\snail.jpg"; //args[0];

            //torchvision.io.DefaultImager = new SkiaImager();

            var img1 = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB);
            var img2 = torchvision.io.read_image(filename, torchvision.io.ImageReadMode.RGB, new SkiaImager());

            var b1 = img1.data<byte>();
            var b2 = img2.data<byte>();

            //for (int i = 0; i < b1.Count; i++) Debug.Assert(b1[i] == b2[i]);

            NewMethod(img1, 1, torchvision.io.DefaultImager);
            NewMethod(img2, 2, new SkiaImager());
        }

        private static void NewMethod(Tensor img, int idx, TorchSharp.torchvision.io.IImager imager)
        {
            var expanded = img.unsqueeze(0);

            Console.WriteLine($"Image has {expanded.shape[1]} colour channels with dimensions {expanded.shape[2]}x{expanded.shape[3]}");

            //var transformed = torchvision.transforms.Compose(
            //    torchvision.transforms.Invert()
            //    ).forward(img);

            var transformed = torchvision.transforms.functional.vflip(expanded);

            torchvision.io.write_image(transformed.squeeze(), $@"d:\repos\niklasgustafsson\TorchSharp\image_transformed_vflip{idx}.png", torchvision.io.ImageFormat.PNG, imager);
        }
    }
}
