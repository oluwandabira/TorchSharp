// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using static TorchSharp.torch;

// A number of implementation details in this file have been translated from the Python version or torchvision,
// largely located in the files found in this folder:
//
// https://github.com/pytorch/vision/tree/993325dd82567f5d4f28ccb321e3a9a16984d2d8/torchvision/transforms
//
// The origin has the following copyright notice and license:
//
// https://github.com/pytorch/vision/blob/master/LICENSE
//

namespace TorchSharp.torchvision
{
    public static partial class transforms
    {
        public static partial class functional
        {

            /// <summary>
            /// Adjust the brightness of an image.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="brightness_factor">
            /// How much to adjust the brightness. Can be any non negative number.
            /// 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
            /// </param>
            /// <returns></returns>
            public static Tensor adjust_brightness(Tensor img, double brightness_factor)
            {
                var scope = NewDisposeScope();

                if (brightness_factor == 1.0)
                    // Special case -- no change.
                    return img.alias();

                using var zeros = torch.zeros_like(img);
                return Blend(img, zeros, brightness_factor).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Adjust the contrast of the image.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="contrast_factor">
            /// How much to adjust the contrast. Can be any non-negative number.
            /// 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
            /// </param>
            /// <returns></returns>
            public static Tensor adjust_contrast(Tensor img, double contrast_factor)
            {
                var scope = NewDisposeScope();

                if (contrast_factor == 1.0)
                    // Special case -- no change.
                    return img.alias();

                var dtype = torch.is_floating_point(img) ? img.dtype : torch.float32;
                using var t0 = transforms.functional.rgb_to_grayscale(img);
                using var t1 = t0.to_type(dtype);
                using var mean = torch.mean(t1, new long[] { -3, -2, -1 }, keepDimension: true);

                return Blend(img, mean, contrast_factor).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Perform gamma correction on an image.
            ///
            /// See: https://en.wikipedia.org/wiki/Gamma_correction
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="gamma">
            /// Non negative real number.
            /// gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            /// </param>
            /// <param name="gain">The constant multiplier in the gamma correction equation.</param>
            /// <returns></returns>
            public static Tensor adjust_gamma(Tensor img, double gamma, double gain = 1.0)
            {
                var scope = NewDisposeScope();

                var dtype = img.dtype;
                if (!torch.is_floating_point(img)) {
                    img = transforms.functional.convert_image_dtype(img, torch.float32);
                } else {
                    img = img.alias();
                }

                using var t0 = img.pow(gamma);
                using var t1 = gain * t0;
                using var t2 = t1.clamp(0, 1);

                return convert_image_dtype(t2, dtype).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Adjust the hue of an image.
            /// The image hue is adjusted by converting the image to HSV and cyclically shifting the intensities in the hue channel(H).
            /// The image is then converted back to original image mode.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="hue_factor">
            /// How much to shift the hue channel. 0 means no shift in hue.
            /// Hue is often defined in degrees, with 360 being a full turn on the color wheel.
            /// In this library, 1.0 is by default a full turn, which means that 0.5 and -0.5 give complete reversal of
            /// the hue channel in HSV space in positive and negative direction respectively.
            /// </param>
            /// <param name="degrees">Whether the hue factor is measured in degrees. If true, 360 is a full turn.</param>
            /// <returns></returns>
            /// <remarks>
            /// Unlike Pytorch, TorchSharp will allow the hue_factor to lie outside the range [-0.5,0.5].
            /// A factor of 0.75 has the same effect as -.25
            /// Note that adjusting the hue is a very expensive operation, and may therefore not be suitable as a method
            /// for data augmentation when training speed is important.
            /// </remarks>
            public static Tensor adjust_hue(Tensor img, double hue_factor, bool degrees = false)
            {
                var scope = NewDisposeScope();

                if (hue_factor == 0.0)
                    // Special case -- no change.
                    return img.alias();

                if (img.shape.Length < 4 || img.shape[img.shape.Length - 3] == 1)
                    // Grayscale, or not a batch of images. Nothing to do.
                    return img.alias();

                if (degrees)
                    hue_factor = hue_factor / 360.0;

                var res = THSVision_AdjustHue(img.Handle, hue_factor);
                if (res == IntPtr.Zero) { CheckForErrors(); }
                return new Tensor(res).MoveToOuterDisposeScope();
            }

            /* Tensor THSVision_AdjustHue(const Tensor i, const double hue_factor) */
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSVision_AdjustHue(IntPtr img, double hue_factor);


            /// <summary>
            /// Adjust the color saturation of an image.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="saturation_factor">
            /// How much to adjust the saturation. 0 will give a black and white image, 1 will give the original image
            /// while 2 will enhance the saturation by a factor of 2.
            /// </param>
            /// <returns></returns>
            public static Tensor adjust_saturation(Tensor img, double saturation_factor)
            {
                if (saturation_factor == 1.0)
                    // Special case -- no change.
                    return img.alias();

                var scope = NewDisposeScope();

                using var t0 = transforms.functional.rgb_to_grayscale(img);
                return Blend(img, t0, saturation_factor).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Adjust the sharpness of the image.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="sharpness">
            /// How much to adjust the sharpness. Can be any non negative number.
            /// 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
            /// </param>
            /// <returns></returns>
            public static Tensor adjust_sharpness(Tensor img, double sharpness)
            {
                if (img.shape[img.shape.Length - 1] <= 2 || img.shape[img.shape.Length - 2] <= 2)
                    return img.alias();

                var scope = NewDisposeScope();

                using var t0 = BlurredDegenerateImage(img);
                return Blend(img, t0, sharpness).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Apply affine transformation on the image keeping image center invariant.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="shear">Shear angle value in degrees between -180 to 180, clockwise direction. </param>
            /// <param name="angle">Rotation angle in degrees between -180 and 180, clockwise direction</param>
            /// <param name="translate">Horizontal and vertical translations (post-rotation translation)</param>
            /// <param name="scale">Overall scale</param>
            /// <param name="interpolation">Desired interpolation.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed image.</param>
            /// <returns></returns>
            public static Tensor affine(Tensor img, IList<float> shear = null, float angle = 0.0f, IList<int> translate = null, float scale = 1.0f, InterpolationMode interpolation = InterpolationMode.Nearest, float? fill = null)
            {
                IList<float> fills = (fill.HasValue) ? new float[] { fill.Value } : null;

                if (translate == null) {
                    translate = new int[] { 0, 0 };
                }

                if (shear == null) {
                    shear = new float[] { 0.0f, 0.0f };
                }

                if (shear.Count == 1) {
                    shear = new float[] { shear[0], shear[0] };
                }

                var scope = NewDisposeScope();

                var matrix = GetInverseAffineMatrix((0.0f, 0.0f), angle, (translate[0], translate[1]), scale, (shear[0], shear[1]));

                var dtype = torch.is_floating_point(img) ? img.dtype : ScalarType.Float32;

                using var t0 = torch.tensor(matrix, dtype: dtype, device: img.device);
                using var theta = t0.reshape(1, 2, 3);

                var end_ = img.shape.Length;

                using var grid = GenerateAffineGrid(theta, img.shape[end_ - 1], img.shape[end_ - 2], img.shape[end_ - 1], img.shape[end_ - 2]);
                return ApplyGridTransform(img, grid, InterpolationMode.Nearest, fill: fills).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Apply affine transformation on the image keeping image center invariant.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="img">An image tensor.</param>
            /// <param name="shear">Shear angle value in degrees between -180 to 180, clockwise direction. </param>
            /// <param name="angle">Rotation angle in degrees between -180 and 180, clockwise direction</param>
            /// <param name="translate">Horizontal and vertical translations (post-rotation translation)</param>
            /// <param name="scale">Overall scale</param>
            /// <param name="interpolation">Desired interpolation.</param>
            /// <param name="fill">Pixel fill value for the area outside the transformed image.</param>
            /// <returns></returns>
            public static Tensor affine(Tensor img, float shear, float angle = 0.0f, IList<int> translate = null, float scale = 1.0f, InterpolationMode interpolation = InterpolationMode.Nearest, float? fill = null)
            {
                return affine(img, new float[] { shear, 0.0f }, angle, translate, scale, interpolation, fill);
            }

            /// <summary>
            /// Maximize contrast of an image by remapping its pixels per channel so that the lowest becomes black and the lightest becomes white.
            /// </summary>
            /// <param name="input"></param>
            /// <returns></returns>
            public static Tensor autocontrast(Tensor input)
            {
                var bound = input.IsIntegral() ? 255.0f : 1.0f;
                var dtype = input.IsIntegral() ? ScalarType.Float32 : input.dtype;

                var scope = NewDisposeScope();

                using var t0 = input.amin(new long[] { -2, -1 }, keepDim: true);
                using var t1 = input.amax(new long[] { -2, -1 }, keepDim: true);

                using var minimum = t0.to(dtype);
                using var maximum = t1.to(dtype);

                using var t2 = (minimum == maximum);
                var t3 = t2.nonzero_as_list();
                var eq_idxs = t3[0];

                using var t4 = minimum.index_put_(0, eq_idxs);
                using var t5 = maximum.index_put_(bound, eq_idxs);

                using var t6 = (maximum - minimum);
                using var t7 = torch.tensor(bound, float32);

                using var scale = t7 / t6;

                using var t8 = (input - minimum);
                using var t9 = t8 * scale;
                using var t10 = t9.clamp(0, bound);

                return t10.to(input.dtype).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Crops the given image at the center. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than
            /// output size along any edge, image is padded with 0 and then center cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="height">The height of the crop box.</param>
            /// <param name="width">The width of the crop box.</param>
            /// <returns></returns>
            public static Tensor center_crop(Tensor input, int height, int width)
            {
                var hoffset = input.Dimensions - 2;
                var iHeight = input.shape[hoffset];
                var iWidth = input.shape[hoffset + 1];

                var top = (int)(iHeight - height) / 2;
                var left = (int)(iWidth - width) / 2;

                var scope = NewDisposeScope();

                return input.crop(top, left, height, width).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Crops the given image at the center. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than
            /// output size along any edge, image is padded with 0 and then center cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="size">The size of the crop box.</param>
            /// <returns></returns>
            public static Tensor center_crop(Tensor input, int size) => center_crop(input, size, size);

            /// <summary>
            /// Convert a tensor image to the given dtype and scale the values accordingly
            /// </summary>
            public static Tensor convert_image_dtype(Tensor image, ScalarType dtype = ScalarType.Float32)
            {
                if (image.dtype == dtype)
                    return image.alias();

                var scope = NewDisposeScope();

                var output_max = MaxValue(dtype);

                if (torch.is_floating_point(image)) {

                    if (torch.is_floating_point(dtype)) {
                        return image.to_type(dtype);
                    }

                    if ((image.dtype == torch.float32 && (dtype == torch.int32 || dtype == torch.int64)) ||
                        (image.dtype == torch.float64 && dtype == torch.int64)) {
                        throw new ArgumentException($"The cast from {image.dtype} to {dtype} cannot be performed safely.");
                    }

                    var eps = 1e-3;
                    using var result = image.mul(output_max + 1.0 - eps);
                    return result.to_type(dtype).MoveToOuterDisposeScope();

                } else {
                    // Integer to floating point.

                    var input_max = MaxValue(image.dtype);

                    if (torch.is_floating_point(dtype)) {
                        using var t0 = image.to_type(dtype);
                        return (t0 / input_max).MoveToOuterDisposeScope();
                    }

                    if (input_max > output_max) {
                        var factor = (input_max + 1) / (output_max + 1);
                        using var t0 = torch.div(image, factor);
                        return t0.to_type(dtype).MoveToOuterDisposeScope();
                    } else {
                        var factor = (output_max + 1) / (input_max + 1);
                        using var t0 = image.to_type(dtype);
                        return (t0 * factor).MoveToOuterDisposeScope();
                    }
                }
            }

            /// <summary>
            /// Crop the given image at specified location and output size. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge,
            /// image is padded with 0 and then cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="height">The height of the crop box.</param>
            /// <param name="width">The width of the crop box.</param>
            /// <returns></returns>
            public static Tensor crop(Tensor input, int top, int left, int height, int width)
            {
                var scope = NewDisposeScope();
                return input.crop(top, left, height, width).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Crop the given image at specified location and output size. The image is expected to have […, H, W] shape,
            /// where … means an arbitrary number of leading dimensions. If image size is smaller than output size along any edge,
            /// image is padded with 0 and then cropped.
            /// </summary>
            /// <param name="input">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="size">The size of the crop box.</param>
            /// <returns></returns>
            public static Tensor crop(Tensor input, int top, int left, int size)
            {
                return crop(input, top, left, size, size);
            }

            /// <summary>
            /// Equalize the histogram of an image by applying a non-linear mapping to the input in order to create a uniform distribution of grayscale values in the output.
            /// </summary>
            /// <param name="input">The image tensor</param>
            /// <returns></returns>
            public static Tensor equalize(Tensor input)
            {
                if (input.dtype != ScalarType.Byte)
                    throw new ArgumentException($"equalize() requires a byte image, but the type of the argument is {input.dtype}.");

                var scope = NewDisposeScope();

                if (input.ndim == 3) {
                    return EqualizeSingleImage(input).MoveToOuterDisposeScope();
                }

                var images = Enumerable.Range(0, (int)input.shape[0]).Select(i => EqualizeSingleImage(input[i])).ToList();
                var result = torch.stack(images);
                foreach (var img in images) { img.Dispose(); }
                return result.MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Erase the input Tensor Image with given value.
            /// </summary>
            /// <param name="img">The input tensor</param>
            /// <param name="top">Vertical component of the top left corner of the erased region.</param>
            /// <param name="left">Horizontal component of the top left corner of the erased region.</param>
            /// <param name="height">The height of the erased region.</param>
            /// <param name="width">The width of the erased region.</param>
            /// <param name="value">Erasing value.</param>
            /// <param name="inplace">For in-place operations.</param>
            /// <returns></returns>
            public static Tensor erase(Tensor img, int top, int left, int height, int width, Tensor value, bool inplace = false)
            {
                var scope = NewDisposeScope();

                if (!inplace) {
                    using var t0 = img.clone();
                    return t0.index_put_(value, new TensorIndex[] { TensorIndex.Ellipsis, (top, top + height), (left, left + width) }).MoveToOuterDisposeScope();
                } else {
                    return img.index_put_(value, new TensorIndex[] { TensorIndex.Ellipsis, (top, top + height), (left, left + width) }).MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, IList<long> kernelSize, IList<float> sigma)
            {
                var dtype = TensorExtensionMethods.IsIntegral(input.dtype) ? ScalarType.Float32 : input.dtype;

                if (kernelSize.Count == 1) {
                    kernelSize = new long[] { kernelSize[0], kernelSize[0] };
                }

                if (sigma == null) {
                    sigma = new float[] {
                        0.3f * ((kernelSize[0] - 1) * 0.5f - 1) + 0.8f,
                        0.3f * ((kernelSize[1] - 1) * 0.5f - 1) + 0.8f,
                    };
                } else if (sigma.Count == 1) {
                    sigma = new float[] {
                        sigma[0],
                        sigma[0],
                    };
                }

                var scope = NewDisposeScope();

                using var t0 = GetGaussianKernel2d(kernelSize, sigma, dtype, input.device);
                using var kernel = t0.expand(input.shape[input.shape.Length - 3], 1, t0.shape[0], t0.shape[1]);

                using var img0 = SqueezeIn(input, new ScalarType[] { kernel.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

                // The padding needs to be adjusted to make sure that the output is the same size as the input.

                var k0d2 = kernelSize[0] / 2;
                var k1d2 = kernelSize[1] / 2;
                var k0sm1 = kernelSize[0] - 1;
                var k1sm1 = kernelSize[1] - 1;

                var padding = new long[] { k0d2, k1d2, k0sm1 - k0d2, k1sm1 - k1d2 };

                using var img1 = TorchSharp.torchvision.transforms.functional.pad(img0, padding, padding_mode: PaddingModes.Reflect);
                using var img2 = torch.nn.functional.conv2d(img1, kernel, groups: img1.shape[img1.shape.Length - 3]);

                return SqueezeOut(img2, needCast, needSqueeze, out_dtype).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, long kernelSize, float sigma)
            {
                return gaussian_blur(input, new long[] { kernelSize, kernelSize }, new float[] { sigma });
            }

            /// <summary>
            /// Performs Gaussian blurring on the image by given kernel.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <returns></returns>
            public static Tensor gaussian_blur(Tensor input, long kernelHeight, long kernelWidth, float sigma_x, float sigma_y)
            {
                return gaussian_blur(input, new long[] { kernelHeight, kernelWidth }, new float[] { sigma_x, sigma_y });
            }

            /// <summary>
            /// Horizontally flip the given image.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <returns></returns>
            public static Tensor hflip(Tensor input) => input.flip(-1);

            /// <summary>
            /// Invert the colors of an RGB/grayscale image.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <returns></returns>
            public static Tensor invert(Tensor input)
            {
                var scope = NewDisposeScope();

                using var t0 = -input;
                if (input.IsIntegral()) {
                    return (t0 + 255).MoveToOuterDisposeScope();
                } else {
                    return (t0 + 1.0).MoveToOuterDisposeScope();
                }
            }

            /// <summary>
            /// Normalize a float tensor image with mean and standard deviation.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="means">Sequence of means for each channel.</param>
            /// <param name="stdevs">Sequence of standard deviations for each channel.</param>
            /// <param name="dtype">Bool to make this operation inplace.</param>
            /// <returns></returns>
            public static Tensor normalize(Tensor input, double[] means, double[] stdevs, ScalarType dtype = ScalarType.Float32)
            {
                var scope = NewDisposeScope();

                if (means.Length != stdevs.Length)
                    throw new ArgumentException("means and stdevs must be the same length in call to Normalize");
                if (means.Length != input.shape[1])
                    throw new ArgumentException("The number of channels is not equal to the number of means and standard deviations");

                using var mean = means.ToTensor(new long[] { 1, means.Length, 1, 1 }).to(input.dtype, input.device);     // Assumes NxCxHxW
                using var stdev = stdevs.ToTensor(new long[] { 1, stdevs.Length, 1, 1 }).to(input.dtype, input.device);  // Assumes NxCxHxW
                using var t0 = input - mean;

                return (t0 / stdev).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Pad the given image on all sides with the given “pad” value.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="padding">
            /// Padding on each border. If a single int is provided this is used to pad all borders.
            /// If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            /// If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
            /// </param>
            /// <param name="fill">Pixel fill value for constant fill.</param>
            /// <param name="padding_mode"></param>
            /// <returns></returns>
            public static Tensor pad(Tensor input, long[] padding, double fill = 0, PaddingModes padding_mode = PaddingModes.Constant)
            {
                long[] correctedPad = new long[4];

                switch (padding.Length) {
                case 1:
                    correctedPad = new long[] { padding[0], padding[0], padding[0], padding[0] };
                    break;
                case 2:
                    correctedPad = new long[] { padding[0], padding[0], padding[1], padding[1] };
                    break;
                case 4:
                    correctedPad = new long[] { padding[0], padding[2], padding[1], padding[3] };
                    break;
                }

                return torch.nn.functional.pad(input, correctedPad, padding_mode, fill);
            }

            /// <summary>
            /// Pad the given image on all sides with the given “pad” value.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="padding">
            /// Padding on each border. If a single int is provided this is used to pad all borders.
            /// If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            /// If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
            /// </param>
            /// <param name="fill">Pixel fill value for constant fill.</param>
            /// <param name="padding_mode"></param>
            /// <returns></returns>
            public static Tensor pad(Tensor input, long padding, double fill = 0, PaddingModes padding_mode = PaddingModes.Constant)
            {
                return torch.nn.functional.pad(input, new[] { padding, padding, padding, padding } , padding_mode, fill);
            }

            /// <summary>
            /// Pad the given image on all sides with the given “pad” value.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="padding">
            /// Padding on each border. If a single int is provided this is used to pad all borders.
            /// If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            /// If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
            /// </param>
            /// <param name="fill">Pixel fill value for constant fill.</param>
            /// <param name="padding_mode"></param>
            /// <returns></returns>
            public static Tensor pad(Tensor input, (long, long) padding, double fill = 0, PaddingModes padding_mode = PaddingModes.Constant)
            {
                return torch.nn.functional.pad(input, new[] { padding.Item1, padding.Item1, padding.Item2, padding.Item2 }, padding_mode, fill);
            }

            /// <summary>
            /// Pad the given image on all sides with the given “pad” value.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="padding">
            /// Padding on each border. If a single int is provided this is used to pad all borders.
            /// If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
            /// If a sequence of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
            /// </param>
            /// <param name="fill">Pixel fill value for constant fill.</param>
            /// <param name="padding_mode"></param>
            /// <returns></returns>
            public static Tensor pad(Tensor input, (long, long, long, long) padding, double fill = 0, PaddingModes padding_mode = PaddingModes.Constant)
            {
                return torch.nn.functional.pad(input, new[] { padding.Item1, padding.Item3, padding.Item2, padding.Item4 }, padding_mode, fill);
            }

            /// <summary>
            /// Perform perspective transform of the given image.
            /// The image is expected to have […, H, W] shape, where … means an arbitrary number of leading dimensions.
            /// </summary>
            /// <param name="img">An image tensor</param>
            /// <param name="startpoints">List containing four lists of two integers corresponding to four corners [top-left, top-right, bottom-right, bottom-left] of the original image.</param>
            /// <param name="endpoints">List containing four lists of two integers corresponding to four corners [top-left, top-right, bottom-right, bottom-left] of the transformed image.</param>
            /// <param name="interpolation">Desired interpolation. Only InterpolationMode.Nearest, InterpolationMode.Bilinear are supported. </param>
            /// <param name="fill">Pixel fill value for the area outside the transformed image.</param>
            /// <returns></returns>
            public static Tensor perspective(Tensor img, IList<IList<int>> startpoints, IList<IList<int>> endpoints, InterpolationMode interpolation = InterpolationMode.Bilinear, IList<float> fill = null)
            {
                var scope = NewDisposeScope();

                if (interpolation != InterpolationMode.Nearest && interpolation != InterpolationMode.Bilinear)
                    throw new ArgumentException($"Invalid interpolation mode for 'perspective': {interpolation}. Use 'nearest' or 'bilinear'.");

                var coeffs = GetPerspectiveCoefficients(startpoints, endpoints);

                var _end = img.shape.Length;
                var ow = img.shape[_end - 1];
                var oh = img.shape[_end - 2];

                var dtype = torch.is_floating_point(img) ? img.dtype : ScalarType.Float32;
                using var grid = PerspectiveGrid(coeffs, ow, oh, dtype: dtype, device: img.device);

                return ApplyGridTransform(img, grid, interpolation, fill).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Posterize an image by reducing the number of bits for each color channel.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="bits">The number of high-order bits to keep.</param>
            /// <returns></returns>
            public static Tensor posterize(Tensor input, int bits)
            {
                var scope = NewDisposeScope();

                if (input.dtype != ScalarType.Byte) throw new ArgumentException("Only torch.byte image tensors are supported");
                using var mask = torch.tensor((byte)-(1 << (8 - bits)));
                return (input & mask).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Resize the input image to the given size.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="height"></param>
            /// <param name="width"></param>
            /// <param name="maxSize"></param>
            /// <returns></returns>
            public static Tensor resize(Tensor input, int height, int width, int? maxSize = null)
            {
                // For now, we don't allow any other modes.
                const InterpolationMode interpolation = InterpolationMode.Nearest;

                var hoffset = input.Dimensions - 2;
                var iHeight = input.shape[hoffset];
                var iWidth = input.shape[hoffset + 1];

                if (iHeight == height && iWidth == width)
                    return input;

                var scope = NewDisposeScope();

                var h = height;
                var w = width;

                if (w == -1) {
                    if (maxSize.HasValue && height > maxSize.Value)
                        throw new ArgumentException($"maxSize = {maxSize} must be strictly greater than the requested size for the smaller edge size = {height}");

                    // Only one size was specified -- retain the aspect ratio.
                    if (iHeight < iWidth) {
                        h = height;
                        w = (int)Math.Floor(height * ((double)iWidth / (double)iHeight));
                    } else if (iWidth < iHeight) {
                        w = height;
                        h = (int)Math.Floor(height * ((double)iHeight / (double)iWidth));
                    } else {
                        w = height;
                    }
                }

                if (interpolation != InterpolationMode.Nearest) {
                    throw new NotImplementedException("Interpolation mode != 'Nearest'");
                }

                using var img0 = SqueezeIn(input, new ScalarType[] { ScalarType.Float32, ScalarType.Float64 }, out var needCast, out var needSqueeze, out var dtype);

                using var img1 = torch.nn.functional.interpolate(img0, new long[] { h, w }, mode: interpolation, align_corners: null);

                return SqueezeOut(img1, needCast, needSqueeze, dtype).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Crop the given image and resize it to desired size.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="top">Vertical component of the top left corner of the crop box.</param>
            /// <param name="left">Horizontal component of the top left corner of the crop box.</param>
            /// <param name="height">Height of the crop box.</param>
            /// <param name="width">Width of the crop box.</param>
            /// <param name="newHeight">New height.</param>
            /// <param name="newWidth">New width.</param>
            /// <returns></returns>
            public static Tensor resized_crop(Tensor input, int top, int left, int height, int width, int newHeight, int newWidth)
            {
                var scope = NewDisposeScope();

                using var t0 = crop(input, top, left, height, width);
                return resize(t0, newHeight, newWidth).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Convert RGB image to grayscale version of image.
            /// </summary>
            /// <param name="input">An image tensor.</param>
            /// <param name="num_output_channels">The number of channels of the output image. Value must be 1 or 3.</param>
            /// <returns></returns>
            public static Tensor rgb_to_grayscale(Tensor input, int num_output_channels = 1)
            {
                if (num_output_channels != 1 && num_output_channels != 3)
                    throw new ArgumentException("The number of output channels must be 1 or 3.");

                int cDim = (int)input.Dimensions - 3;
                if (input.shape[cDim] == 1)
                    // Already grayscale...
                    return input.alias();

                var scope = NewDisposeScope();

                var dtype = input.dtype;

                if (!is_floating_point(dtype)) {
                    input = convert_image_dtype(input, torch.float32);
                }

                var rgb = input.unbind(cDim);
                using var img = (rgb[0] * 0.2989 + rgb[1] * 0.587 + rgb[2] * 0.114).unsqueeze(cDim);
                foreach (var c in rgb) { c.Dispose(); }

                var result = num_output_channels == 3 ? img.expand(input.shape) : img.alias();

                if (!is_floating_point(dtype)) {
                    result = convert_image_dtype(result, dtype);
                }

                return result.MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Rotate the image by angle, counter-clockwise.
            /// </summary>
            public static Tensor rotate(Tensor img, float angle, InterpolationMode interpolation = InterpolationMode.Nearest, bool expand = false, (int, int)? center = null, IList<float> fill = null)
            {
                var center_f = (0.0f, 0.0f);

                var scope = NewDisposeScope();

                if (center.HasValue) {
                    var img_size = GetImageSize(img);
                    center_f = (1.0f * (center.Value.Item1 - img_size.Item1 * 0.5f), 1.0f * (center.Value.Item2 - img_size.Item2 * 0.5f));
                }

                var matrix = GetInverseAffineMatrix(center_f, -angle, (0.0f, 0.0f), 1.0f, (0.0f, 0.0f));

                return RotateImage(img, matrix, interpolation, expand, fill).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Solarize an RGB/grayscale image by inverting all pixel values above a threshold.
            /// </summary>
            /// <returns></returns>
            public static Tensor solarize(Tensor input, double threshold)
            {
                var scope = NewDisposeScope();

                using (var inverted = invert(input))
                using (var filter = input < threshold)
                    return torch.where(filter, input, inverted).MoveToOuterDisposeScope();
            }

            /// <summary>
            /// Vertically flip the given image.
            /// </summary>
            public static Tensor vflip(Tensor input) => input.flip(-2);


            //
            // Supporting implementation details.
            //
            private static Tensor RotateImage(Tensor img, IList<float> matrix, InterpolationMode interpolation, bool expand, IList<float> fill)
            {
                var scope = NewDisposeScope();

                var (w, h) = GetImageSize(img);
                var (ow, oh) = expand ? ComputeOutputSize(matrix, w, h) : (w, h);
                var dtype = torch.is_floating_point(img) ? img.dtype : torch.float32;
                using var t0 = torch.tensor(matrix, dtype: dtype, device: img.device);
                using var theta = t0.reshape(1, 2, 3);
                using var grid = GenerateAffineGrid(theta, w, h, ow, oh);

                return ApplyGridTransform(img, grid, interpolation, fill).MoveToOuterDisposeScope();
            }

            private static Tensor Blend(Tensor img1, Tensor img2, double ratio)
            {
                var scope = NewDisposeScope();

                var bound = img1.IsIntegral() ? 255.0 : 1.0;
                using var t0 = img1 * ratio;
                using var t2 = img2 * (1.0 - ratio);
                using var t3 = (t0 + t2);
                using var t4 = t3.clamp(0, bound);
                return t4.to(img2.dtype).MoveToOuterDisposeScope();
            }

            private static Tensor BlurredDegenerateImage(Tensor input)
            {
                var scope = NewDisposeScope();

                var device = input.device;
                var dtype = input.IsIntegral() ? ScalarType.Float32 : input.dtype;
                using var kernel = torch.ones(3, 3, device: device);
                using var t0 = torch.tensor(5.0f);
                kernel[1, 1] = t0;

                using var t1 = kernel.sum();
                using var t2 = kernel / t1;
                using var t3 = t2.expand(input.shape[input.shape.Length - 3], 1, kernel.shape[0], kernel.shape[1]);

                using var t4 = SqueezeIn(input, new ScalarType[] { ScalarType.Float32, ScalarType.Float64 }, out var needCast, out var needSqueeze, out var out_dtype);
                using var t5 = torch.nn.functional.conv2d(t4, t3, groups: t4.shape[t4.shape.Length - 3]);
                using var result_tmp = SqueezeOut(t5, needCast, needSqueeze, out_dtype);

                using var result = input.clone();
                return result.index_put_(result_tmp, TensorIndex.Ellipsis, TensorIndex.Slice(1, -1), TensorIndex.Slice(1, -1)).MoveToOuterDisposeScope();
            }

            private static Tensor GetGaussianKernel1d(long size, float sigma)
            {
                var ksize_half = (size - 1) * 0.5f;
                using var x = torch.linspace(-ksize_half, ksize_half, size);
                using var t0 = x / sigma;
                using var t1 = -t0;
                using var t2 = t1.pow(2);

                using var pdf = t2 * 0.5f;
                using var sum = pdf.sum();

                return pdf / sum;
            }

            private static Tensor GetGaussianKernel2d(IList<long> kernelSize, IList<float> sigma, ScalarType dtype, torch.Device device)
            {
                var scope = NewDisposeScope();

                using var tX1 = GetGaussianKernel1d(kernelSize[0], sigma[0]);
                using var tX2 = tX1.to(dtype, device);
                using var kernel_X = tX2[TensorIndex.None, TensorIndex.Slice()];

                using var tY1 = GetGaussianKernel1d(kernelSize[1], sigma[1]);
                using var tY2 = tY1.to(dtype, device);
                using var kernel_Y = tY2[TensorIndex.Slice(), TensorIndex.None];

                return kernel_Y.mm(kernel_X).MoveToOuterDisposeScope();
            }

            // EXPORT_API(Tensor) THSVision_ApplyGridTransform(Tensor img, Tensor grid, const int8_t m, const float* fill, const int64_t fill_length);
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSVision_ApplyGridTransform(IntPtr img, IntPtr grid, sbyte mode, IntPtr fill, long fill_length);

            private static Tensor ApplyGridTransform(Tensor img, Tensor grid, InterpolationMode mode, IList<float> fill = null)
            {
                var scope = NewDisposeScope();

                img = SqueezeIn(img, new ScalarType[] { grid.dtype }, out var needCast, out var needSqueeze, out var out_dtype);

                var fillLength = (fill != null) ? fill.Count : 0;
                var fillArray = (fill != null) ? fill.ToArray() : null;

                unsafe {
                    fixed (float* pfill = fillArray) {
                        var res = THSVision_ApplyGridTransform(img.Handle, grid.Handle, (sbyte)mode, (IntPtr)pfill, fillLength);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        img = new Tensor(res);
                    }
                }

                img = SqueezeOut(img, needCast, needSqueeze, out_dtype);
                return img.MoveToOuterDisposeScope();
            }

            /* Tensor THSVision_GenerateAffineGrid(Tensor theta, const int64_t w, const int64_t h, const int64_t ow, const int64_t oh); */
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSVision_GenerateAffineGrid(IntPtr theta, long w, long h, long ow, long oh);


            private static Tensor GenerateAffineGrid(Tensor theta, long w, long h, long ow, long oh)
            {
                var img = THSVision_GenerateAffineGrid(theta.Handle, w, h, ow, oh);
                if (img == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(img);
            }

            /* Tensor THSVision_ComputeOutputSize(const float* matrix, const int64_t matrix_length, const int64_t w, const int64_t h); */
            [DllImport("LibTorchSharp")]
            extern static void THSVision_ComputeOutputSize(IntPtr matrix, long matrix_length, long w, long h, out int first, out int second);

            private static (int, int) ComputeOutputSize(IList<float> matrix, long w, long h)
            {
                if (matrix == null)
                    throw new ArgumentNullException("matrix");

                var fillLength = matrix.Count;
                var fillArray = matrix.ToArray();

                unsafe {
                    fixed (float* pfill = fillArray) {
                        int first, second;
                        THSVision_ComputeOutputSize((IntPtr)pfill, fillLength, w, h, out first, out second);
                        torch.CheckForErrors();
                        return (first, second);
                    }
                }
            }

            private static IList<float> GetInverseAffineMatrix((float, float) center, float angle, (float, float) translate, float scale, (float, float) shear)
            {
                // Convert to radians.
                var rot = angle * MathF.PI / 180.0f;
                var sx = shear.Item1 * MathF.PI / 180.0f;
                var sy = shear.Item2 * MathF.PI / 180.0f;

                var (cx, cy) = center;
                var (tx, ty) = translate;

                var a = MathF.Cos(rot - sy) / MathF.Cos(sy);
                var b = -MathF.Cos(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) - MathF.Sin(rot);
                var c = MathF.Sin(rot - sy) / MathF.Cos(sy);
                var d = -MathF.Sin(rot - sy) * MathF.Tan(sx) / MathF.Cos(sy) + MathF.Cos(rot);

                var matrix = (new float[] { d, -b, 0.0f, -c, a, 0.0f }).Select(x => x / scale).ToArray();

                matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty);
                matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty);

                matrix[2] += cx;
                matrix[5] += cy;

                return matrix;
            }

            private static (long, long) GetImageSize(Tensor img)
            {
                var hOffset = img.shape.Length - 2;
                return (img.shape[hOffset + 1], img.shape[hOffset]);
            }

            /* Tensor THSVision_PerspectiveGrid(const float* coeffs, const int64_t coeffs_length, const int64_t ow, const int64_t oh, const int8_t scalar_type, const int device_type, const int device_index); */
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSVision_PerspectiveGrid(IntPtr coeffs, long coeffs_length, long ow, long oh, sbyte dtype, int device_type, int device_index);

            private static Tensor PerspectiveGrid(IList<float> coeffs, long ow, long oh, ScalarType dtype, Device device)
            {
                if (coeffs == null)
                    throw new ArgumentNullException("coeffs");
                var fillLength = coeffs.Count;
                var fillArray = coeffs.ToArray();

                unsafe {
                    fixed (float* pfill = fillArray) {
                        var res = THSVision_PerspectiveGrid((IntPtr)pfill, fillLength, ow, oh, (sbyte)dtype, (int)device.type, device.index);
                        if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                        return new Tensor(res);
                    }
                }
            }

            private static IList<float> GetPerspectiveCoefficients(IList<IList<int>> startpoints, IList<IList<int>> endpoints)
            {
                using var a_matrix = torch.zeros(2 * startpoints.Count, 8, dtype: torch.float32);

                for (int i = 0; i < startpoints.Count; i++) {
                    var p1 = endpoints[i];
                    var p2 = startpoints[i];
                    a_matrix[2 * i, TensorIndex.Colon] = torch.tensor(new int[] { p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1] }, dtype: torch.float32);
                    a_matrix[2 * i + 1, TensorIndex.Colon] = torch.tensor(new int[] { 0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1] }, dtype: torch.float32);
                }

                using var b_matrix = torch.tensor(startpoints.SelectMany(sp => sp).ToArray(), dtype: torch.float32).view(8);

                var a_str = a_matrix.ToString(TensorStringStyle.Julia);
                var b_str = b_matrix.ToString(TensorStringStyle.Julia);

                var t0 = torch.linalg.lstsq(a_matrix, b_matrix);

                var t1 = t0.Solution.data<float>().ToArray();

                t0.Solution.Dispose();
                t0.Rank.Dispose();
                t0.Residuals.Dispose();
                t0.SingularValues.Dispose();

                return t1;
            }

            private static Tensor EqualizeSingleImage(Tensor img)
            {
                var scope = NewDisposeScope();
                var channels = new Tensor[] { img[0], img[1], img[2] };

                var t0 = channels.Select(c => ScaleChannel(c)).ToList();
                var t1 = torch.stack(t0);
                foreach (var c in channels) { c.Dispose(); }
                foreach (var c in t0) { c.Dispose(); }
                return t1.MoveToOuterDisposeScope();
            }

            /* Tensor THSVision_ScaleChannel(Tensor ic); */
            [DllImport("LibTorchSharp")]
            extern static IntPtr THSVision_ScaleChannel(IntPtr img);


            private static Tensor ScaleChannel(Tensor img_chan)
            {
                var res = THSVision_ScaleChannel(img_chan.Handle);
                if (res == IntPtr.Zero) { torch.CheckForErrors(); }
                return new Tensor(res);
            }

            internal static Tensor SqueezeIn(Tensor img, IList<ScalarType> req_dtypes, out bool needCast, out bool needSqueeze, out ScalarType dtype)
            {
                var scope = NewDisposeScope();
                needSqueeze = false;

                if (img.Dimensions < 4) {
                    img = img.unsqueeze(0);
                    needSqueeze = true;
                } else {
                    img = img.alias();
                }

                dtype = img.dtype;
                needCast = false;

                if (!req_dtypes.Contains(dtype)) {
                    needCast = true;
                    var t0 = img.to_type(req_dtypes[0]);
                    img.Dispose();
                    img = t0;
                } else {
                    img = img.alias();
                }

                return img.MoveToOuterDisposeScope();
            }

            internal static Tensor SqueezeOut(Tensor img, bool needCast, bool needSqueeze, ScalarType dtype)
            {
                var scope = NewDisposeScope();
                if (needSqueeze) {
                    img = img.squeeze(0);
                } else {
                    img = img.alias();
                }

                if (needCast) {
                    if (TensorExtensionMethods.IsIntegral(dtype)) {
                        var t0 = img.round();
                        img.Dispose();
                        img = t0;
                    }

                    var t1 = img.to_type(dtype);
                    img.Dispose();
                    img = t1;
                }

                return img.MoveToOuterDisposeScope();
            }

            private static long MaxValue(ScalarType dtype)
            {
                switch (dtype) {
                case ScalarType.Byte:
                    return byte.MaxValue;
                case ScalarType.Int8:
                    return sbyte.MaxValue;
                case ScalarType.Int16:
                    return short.MaxValue;
                case ScalarType.Int32:
                    return int.MaxValue;
                case ScalarType.Int64:
                    return long.MaxValue;
                }

                return 0L;
            }

        }
    }
}
