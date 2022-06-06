// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SkiaSharp;
using static TorchSharp.torch;
using TorchSharp.torchvision;
using static TorchSharp.torchvision.io;

using System.Runtime.InteropServices;

namespace TorchSharp.Examples
{
    public class ImageTransforms
    {
        internal static void Main(string[] args)
        {
            var images = new string[] {
                //
                // Find some PNG (or JPEG, etc.) files, download them, and then put their file paths here.
                //
                @"C:\Users\niklasg\Downloads\desert.jpg",
                @"C:\Users\niklasg\Downloads\waterfall.jpg",
                @"C:\Users\niklasg\Downloads\wintersky.jpg",
            };

            const string outputPathPrefix = /* Add the very first part of your repo path here. */ @"d:\repos\niklasgustafsson" + @"\TorchSharp\output-";
            var tensors = LoadImages(images, 4, 3, 256, 256);

            var first = tensors[0];

            int n = 0;

            // First, use the transform version.

            var transform = torchvision.transforms.Compose(
                torchvision.transforms.ConvertImageDType(ScalarType.Float32),
                //torchvision.transforms.ColorJitter(.5f, .5f, .5f, .25f),
                torchvision.transforms.ConvertImageDType(ScalarType.Byte),
                torchvision.transforms.Resize(256, 256)
                );

            var second = transform.forward(first);

            for (; n < second.shape[0]; n++) {

                var image = second[n]; // CxHxW
                var channels = image.shape[0];

                using (var stream = File.OpenWrite(outputPathPrefix + n + ".png")) {
                    var bitmap = GetBitmapFromBytes(image.data<byte>().ToArray(), 256, 256, channels == 1 ? SKColorType.Gray8 : SKColorType.Bgra8888);
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }

            // Then the functional API version.

            second = torchvision.transforms.functional.convert_image_dtype(first);
            // Have to do this to make sure that everything's in the right format before saving.
            second = torchvision.transforms.functional.convert_image_dtype(second, dtype: ScalarType.Byte);
            second = torchvision.transforms.functional.equalize(second);
            second = torchvision.transforms.functional.resize(second, 256, 256);

            for (n = 0; n < second.shape[0]; n++) {

                var image = second[n]; // CxHxW
                var channels = image.shape[0];

                using (var stream = File.OpenWrite(outputPathPrefix + (n + first.shape[0]) + ".png")) {
                    var bitmap = GetBitmapFromBytes(image.data<byte>().ToArray(), 256, 256, channels == 1 ? SKColorType.Gray8 : SKColorType.Bgra8888);
                    bitmap.Encode(stream, SKEncodedImageFormat.Png, 100);
                }
            }
        }

        private static List<Tensor> LoadImages(IList<string> images, int batchSize, int channels, int height, int width)
        {
            List<Tensor> tensors = new List<Tensor>();

            var imgSize = channels * height * width;
            bool shuffle = false;

            Random rnd = new Random();
            var indices = !shuffle ?
                Enumerable.Range(0, images.Count).ToArray() :
                Enumerable.Range(0, images.Count).OrderBy(c => rnd.Next()).ToArray();


            // Go through the data and create tensors
            for (var i = 0; i < images.Count;) {

                var take = Math.Min(batchSize, Math.Max(0, images.Count - i));

                if (take < 1) break;

                var dataTensor = torch.zeros(new long[] { take, imgSize }, ScalarType.Byte);

                // Take
                for (var j = 0; j < take; j++) {
                    var idx = indices[i++];
                    var lblStart = idx * (1 + imgSize);
                    var imgStart = lblStart + 1;

                    var buffer = File.ReadAllBytes(images[idx]);

                    using (var inputTensor = DecodeImage(buffer, ImageReadMode.RGB)) {

                        Tensor finalized = inputTensor;

                        if (inputTensor.shape[2] != width || inputTensor.shape[1] != height) {
                            finalized = torchvision.transforms.functional.resize(finalized, height, width).reshape(imgSize);
                        } else {
                            finalized = finalized.alias();
                        }

                        dataTensor.index_put_(finalized, TensorIndex.Single(j));
                    }
                }

                tensors.Add(dataTensor.reshape(take, channels, height, width));

                dataTensor.Dispose();
            }

            return tensors;
        }

        private static Tensor DecodeImage(byte[] buffer, ImageReadMode mode = ImageReadMode.UNCHANGED)
        {
            using var scope = NewDisposeScope();

            using var bitmap = SKBitmap.Decode(buffer);

            var result = torch.tensor(bitmap.Bytes).reshape(bitmap.Height, bitmap.Width, - 1).permute(2, 0, 1);

            if (mode == ImageReadMode.UNCHANGED)
                return result.DetatchFromDisposeScope();

            var channels = result.shape[0];

            switch (mode) {

            case ImageReadMode.RGB_ALPHA:
                return torch.tensor(GetBytes(bitmap, mode, false));
            case ImageReadMode.RGB:
                if (channels == 4) {
                    result = result[(0, 3)];
                } else if (channels == 1) {
                    result = torch.row_stack(new Tensor[] { result, result, result });
                }
                break;
            case ImageReadMode.GRAY:
                if (channels == 4) {
                    result = torchvision.transforms.functional.rgb_to_grayscale(result[(1, 3)], 1);
                }
                else if (channels == 3) {
                    result = torchvision.transforms.functional.rgb_to_grayscale(result, 1);
                }
                break;
            default: throw new NotImplementedException();
            }

            return result.DetatchFromDisposeScope();
        }

        private static byte[] GetBytes(SKBitmap bitmap, ImageReadMode mode, bool skipAlpha = true)
        {
            var height = bitmap.Height;
            var width = bitmap.Width;

            var inputBytes = bitmap.Bytes;

            if (bitmap.ColorType == SKColorType.Gray8 && mode == ImageReadMode.GRAY)
                return inputBytes;

            if (bitmap.BytesPerPixel != 4 && bitmap.BytesPerPixel != 1)
                throw new ArgumentException("Conversion only supports grayscale and ARGB");

            var iamgeSize = height * width;

            var channelCount = skipAlpha ? 3 : 4;

            int inputBlue = 0, inputGreen = 0, inputRed = 0, inputAlpha = 0;
            int outputRed = 0, outputGreen = iamgeSize, outputBlue = iamgeSize * 2, outputAlpha = iamgeSize * 3;

            switch (bitmap.ColorType) {
            case SKColorType.Bgra8888:
                inputBlue = 0;
                inputGreen = 1;
                inputRed = 2;
                inputAlpha = 3;
                break;
            case SKColorType.Gray8:
                inputBlue = 0;
                inputGreen = 0;
                inputRed = 0;
                inputAlpha = 0;
                break;
            default:
                throw new NotImplementedException($"Conversion from {bitmap.ColorType} to bytes");
            }
            var outBytes = new byte[channelCount * iamgeSize];

            for (int i = 0, j = 0; i < iamgeSize; i += 1, j += 4) {
                outBytes[outputRed + i] = inputBytes[inputRed + j];
                outBytes[outputGreen + i] = inputBytes[inputGreen + j];
                outBytes[outputBlue + i] = inputBytes[inputBlue + j];
                if (!skipAlpha)
                    outBytes[outputAlpha + i] = inputBytes[inputAlpha + j];
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
            var info = new SKImageInfo(width, height, colorType, SKAlphaType.Unpremul);
            result.InstallPixels(info, gcHandle.AddrOfPinnedObject(), info.RowBytes, delegate { gcHandle.Free(); }, null);

            return result;
        }

         [DllImport("LibTorchSharp")]
         static extern void THSVision_BRGA_RGB(IntPtr inputBytes, IntPtr outBytes, int inputChannelCount, int imageSize);

         [DllImport("LibTorchSharp")]
         static extern void THSVision_BRGA_RGBA(IntPtr inputBytes, IntPtr outBytes, int inputChannelCount, int imageSize);

         [DllImport("LibTorchSharp")]
         static extern void THSVision_RGB_BRGA(IntPtr inputBytes, IntPtr outBytes, int inputChannelCount, int imageSize);
    }
}
