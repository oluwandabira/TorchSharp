using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TorchSharp.torchvision
{
    public static partial class io
    {
        /// <summary>
        /// <cref>Imager</cref> to be used when a <cref>torchvision.io</cref> image method's <c>imager</c> is unspecified.
        /// </summary>
        public static Imager DefaultImager { get; set; } = new SkiaImager();

        /// <summary>
        /// Abstract class providing a generic way to decode and encode images as <cref>Tensor</cref>s.
        /// Used by <cref>torchvision.io</cref> image methods.
        /// </summary>
        public abstract class Imager
        {
            /// <summary>
            /// Reads the contents of an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="image">Image file contents.</param>
            /// <param name="mode">Image read mode.</param>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public abstract Tensor DecodeImage(Stream image, ImageReadMode mode = ImageReadMode.UNCHANGED);

            /// <summary>
            /// Reads the contents of an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="image">Image file contents.</param>
            /// <param name="mode">Image read mode.</param>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
            /// </returns>
            public abstract Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED);

            /// <summary>
            /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into an array of bytes.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="format">Image format.</param>
            /// <returns>The encoded image.</returns>
            public abstract byte[] EncodeImage(Tensor image, ImageFormat format);

            /// <summary>
            /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into an array of bytes.
            /// </summary>
            /// <param name="stream">An output stream.</param>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="format">Image format.</param>
            /// <returns>The encoded image.</returns>
            public abstract void EncodeImage(Stream stream, Tensor image, ImageFormat format);
        }

        /// <summary>
        /// Support for various modes while reading images. Affects the returned <cref>Tensor</cref>'s <c>color_channels</c>.
        /// </summary>
        public enum ImageReadMode
        {
            /// <summary>
            /// Read as is. Returned <cref>Tensor</cref>'s color_channels depend on the <cref>ImageFormat</cref>.
            /// </summary>
            UNCHANGED,
            /// <summary>
            /// Read as grayscale. Return <cref>Tensor</cref> with <c>color_channels = 1 </c>.
            /// </summary>
            GRAY,
            /// <summary>
            /// Read as grayscale with transparency. Return <cref>Tensor</cref> with <c>color_channels = 2 </c>.
            /// </summary>
            GRAY_ALPHA,
            /// <summary>
            /// Read as RGB. Return <cref>Tensor</cref> with <c>color_channels = 3 </c>.
            /// </summary>
            RGB,
            /// <summary>
            /// Read as RGB with transparency. Return <cref>Tensor</cref> with <c>color_channels = 4 </c>.
            /// </summary>
            RGB_ALPHA
        }

        /// <summary>
        /// Reads an image file and returns the result as a <cref>Tensor</cref>.
        /// </summary>
        /// <param name="filename">Path to the image.</param>
        /// <param name="mode">Image read mode.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        /// <returns>
        /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
        /// </returns>
        public static Tensor read_image(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            var imgr = imager ?? DefaultImager;
            using (FileStream stream = File.Open(filename, FileMode.Open)) {
                return imgr.DecodeImage(stream, mode);
        }
        }

        /// <summary>
        /// Asynchronously reads an image file and returns the result as a <cref>Tensor</cref>.
        /// </summary>
        /// <param name="filename">Path to the image.</param>
        /// <param name="mode">Read mode.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        /// <returns>
        /// A task that represents the asynchronous read operation.
        /// The value of the TResult parameter is a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
        /// </returns>
        public static async Task<Tensor> read_image_async(string filename, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            byte[] data;

            using (FileStream stream = File.Open(filename, FileMode.Open)) {
                data = new byte[stream.Length];
                await stream.ReadAsync(data, 0, data.Length);
        }

            var imgr = imager ?? DefaultImager;
            return imgr.DecodeImage(data, mode);
        }

        /// <summary>
        /// Write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
        /// </summary>
        /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
        /// <param name="filename">Path to the file.</param>
        /// <param name="format">Image format.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        public static void write_image(Tensor image, string filename, ImageFormat format, Imager imager = null)
        {
            using (var stream = File.Create(filename)) {
                var imgr = imager ?? DefaultImager;
                imgr.EncodeImage(stream, image, format);
            }
        }

        /// <summary>
        /// Asynchronously write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
        /// </summary>
        /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
        /// <param name="filename">Path to the file.</param>
        /// <param name="format">Image format.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        public static async void write_image_async(Tensor image, string filename, ImageFormat format, Imager imager = null)
        {
            var imgr = imager ?? DefaultImager;
            var data = imgr.EncodeImage(image, format);
            using (FileStream stream = File.Create(filename)) {
                await stream.WriteAsync(data, 0, data.Length);
            }
        }

        /// <summary>
        /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>
        /// into a image <cref>Tensor</cref> buffer.
        /// </summary>
        /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
        /// <param name="format">Image format.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        /// <returns>A one dimensional <c>uint8</c> <cref>Tensor</cref> that contains the raw bytes of <c>image</c> encoded in the provided format.</returns>
        public static Tensor encode_image(Tensor image, ImageFormat format, Imager imager = null)
        {
            var imgr = imager ?? DefaultImager;
            return imgr.EncodeImage(image, format);
        }


        /// <summary>
        /// Decodes an image <cref>Tensor</cref> buffer into a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
        /// </summary>
        /// <param name="image">A one dimensional <c>uint8</c> <cref>Tensor</cref> that contains the raw bytes of an image.</param>
        /// <param name="mode">Decode mode.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        /// <returns><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</returns>
        public static Tensor decode_image(Tensor image, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            var imgr = imager ?? DefaultImager;
            return imgr.DecodeImage(image.bytes.ToArray(), mode);
        }
    }
}
