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
                // They shoudl be square and at least 256x256, preferrably larger
                //
            };

            const string outputPathPrefix = /* Add the very first part of your repo path here. */ @"\TorchSharp\output-";
            var tensors = LoadImages(images, 4, 3, 256, 256);

            var first = tensors[0];

            int n = 0;

            // First, use the transform version.

            var transform = torchvision.transforms.Compose(
                torchvision.transforms.ConvertImageDType(ScalarType.Float32),
                torchvision.transforms.ColorJitter(.5f, .5f, .5f, .25f),
                torchvision.transforms.ConvertImageDType(ScalarType.Byte),
                torchvision.transforms.Resize(256, 256)
                );

            var imager = new torchvision.io.SkiaImager();

            var second = transform.forward(first);

            for (; n < second.shape[0]; n++) {

                var image = second[n]; // CxHxW
                var channels = image.shape[0];

                torchvision.io.write_image(image, outputPathPrefix + n + ".png", ImageFormat.Png);
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
                    imager.EncodeImage(stream, image, ImageFormat.Png);
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

            var imager = new torchvision.io.SkiaImager();

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

                    using (var inputTensor = torchvision.io.read_image(images[idx], ImageReadMode.RGB)) {

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
    }
}
