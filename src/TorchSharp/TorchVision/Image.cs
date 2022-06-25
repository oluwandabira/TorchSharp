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
        public static Imager DefaultImager { get; set; } = new NotImplementedImager();

        /// <summary>
        /// Abstract class providing a generic way to decode and encode images as <cref>Tensor</cref>s.
        /// Used by <cref>torchvision.io</cref> image methods.
        /// </summary>
        public abstract class Imager
        {
            /// <summary>
            /// Decode the contents of an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="image">Image file contents.</param>
            /// <param name="mode">Image read mode.</param>
            /// <remarks>
            /// The image format is detected from image file contents.
            /// </remarks>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
            /// </returns>
            public abstract Tensor DecodeImage(byte[] image, ImageReadMode mode = ImageReadMode.UNCHANGED);
            /// <summary>
            /// Decode the contents of an image file and returns the result as a <cref>Tensor</cref>.
            /// </summary>
            /// <param name="image">Image file contents.</param>
            /// <param name="mode">Image read mode.</param>
            /// <remarks>
            /// The image format is detected from image file contents.
            /// </remarks>
            /// <returns>
            /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.
            /// </returns>
            public abstract Tensor DecodeImage(Stream image, ImageReadMode mode = ImageReadMode.UNCHANGED);

            /// <summary>
            /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into an array of bytes.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="format">Image format.</param>
            /// <returns>The encoded image.</returns>
            public abstract byte[] EncodeImage(Tensor image, ImageFormat format);
            /// <summary>
            /// Encodes a <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a stream.
            /// </summary>
            /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
            /// <param name="format">Image format.</param>
            /// <param name="stream">Stream to write to.</param>
            /// <returns>The encoded image.</returns>
            public abstract void EncodeImage(Tensor image, ImageFormat format, Stream stream);
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
            return (imager ?? DefaultImager).DecodeImage(File.ReadAllBytes(filename), mode);
        }

        /// <summary>
        /// Reads an image file and returns the result as a <cref>Tensor</cref>.
        /// </summary>
        /// <param name="stream">Stream to read from.</param>
        /// <param name="mode">Image read mode.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        /// <returns>
        /// <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> and <c>dtype = uint8</c>.
        /// </returns>
        public static Tensor read_image(Stream stream, ImageReadMode mode = ImageReadMode.UNCHANGED, Imager imager = null)
        {
            return (imager ?? DefaultImager).DecodeImage(stream, mode);
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

            return (imager ?? DefaultImager).DecodeImage(data, mode);
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
            File.WriteAllBytes(filename, (imager ?? DefaultImager).EncodeImage(image, format));
        }

        /// <summary>
        /// Write a image <cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c> into a file.
        /// </summary>
        /// <param name="image"><cref>Tensor</cref> with <c>shape = [color_channels, image_height, image_width]</c>.</param>
        /// <param name="stream">Stream to write to.</param>
        /// <param name="format">Image format.</param>
        /// <param name="imager"><cref>Imager</cref> to be use. Will use <cref>DefaultImager</cref> if null.</param>
        public static void write_image(Tensor image, Stream stream, ImageFormat format, Imager imager = null)
        {
            (imager ?? DefaultImager).EncodeImage(image, format, stream);
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
            var data = (imager ?? DefaultImager).EncodeImage(image, format);
            using (FileStream stream = File.Open(filename, FileMode.OpenOrCreate)) {
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
            return (imager ?? DefaultImager).EncodeImage(image, format);
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
            return (imager ?? DefaultImager).DecodeImage(image.bytes.ToArray(), mode);
        }
    }
}