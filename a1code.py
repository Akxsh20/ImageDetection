### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math
import cv2
from cv2 import imread
import numpy as np
from skimage import io


def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    # YOUR CODE HERE
    img = imread(img_path)
    out = img.astype(np.float32) / 255.0

    return out


def print_stats(image):
    """Prints the height, width and number of channels in an image.

    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).

    Returns: none

    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        height, width = image.shape
        print(f"Height: {height}, Width: {width}, Color: grey scale ")
    else:
        height, width, color = image.shape
        print(f"Height: {height}, Width: {width}, Color:{color}")
    # YOUR CODE HERE
    return None


def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index
        start_col (int): The starting column index
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    out = image[start_row : start_row + num_rows, start_col : start_col + num_cols]

    ### YOUR CODE HERE

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    ### YOUR CODE HERE
    out = factor * (image - 0.5) + 0.5
    # Ensure the values remain within the valid range [0.0, 1.0]
    out = np.clip(out, 0.0, 1.0)
    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    height, width = input_image.shape[:2]
    new_height = int(height * output_cols)
    new_width = int(width * output_rows)
    # interpolation
    out = cv2.resize(
        input_image, (new_height, new_width), interpolation=cv2.INTER_NEAREST
    )

    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale.
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # out = np.mean(input_image, axis=2)
    return out


def binary(img, threshold):

    im_bw = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)[1]

    # Apply adaptive thresholding

    return im_bw


def conv2D(image, kernel, stride=1):
    """Convolution of a 2D image with a 2D kernel.
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    Hi, Wi, Ci = image.shape
    Hk, Wk = kernel.shape
    pad_height, pad_width = Hk // 2, Wk // 2

    padded_image = np.pad(
        image,
        ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    Ho = (Hi + 2 * pad_height - Hk) // stride + 1
    Wo = (Wi + 2 * pad_width - Wk) // stride + 1

    # Initialize the output array
    out = np.zeros((Ho, Wo, Ci))
    flipped_kernel = np.flip(kernel)

    # Perform the convolution
    for c in range(Ci):
        for i in range(Ho):
            for j in range(Wo):
                region = padded_image[i : i + Hk, j : j + Wk, c]
                out[i, j, c] = np.sum(region * flipped_kernel)
    ### YOUR CODE HERE
    if out.shape[2] == 1:
        out = out[:, :, 0]

    return out


def test_conv2D():
    """A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.

    Returns:
        None
    """

    # Test code written by
    # Simple convolution kernel.
    kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert (
        np.max(test_output - expected_output) < 1e-10
    ), "Your solution is not correct."
    return test_output


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    return out


def gauss2D(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.

    Args:
        size: filter height and width
        sigma: std deviation of Gaussian

    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()


def corr(image, kernel):
    """Cross correlation of an image with a 2D kernel

    Args:
        image: numpy array of shape (Hi, Wi) or (Hi, Wi, 3)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    if len(image.shape) == 3:  # RGB image
        Hi, Wi, num_channels = image.shape
    else:  # grayscale image
        Hi, Wi = image.shape
        num_channels = 1

    Hk, Wk = kernel.shape

    # Pad the image to handle boundary pixels
    pad_height = Hk // 2
    pad_width = Wk // 2

    if num_channels == 1:
        padded_image = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width)), mode="constant"
        )
    else:
        padded_image = np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
            mode="constant",
        )

    # Initialize the output array
    if num_channels == 1:
        out = np.zeros((Hi, Wi))
    else:
        out = np.zeros((Hi, Wi, num_channels))

    # Iterate over each pixel in the image
    for i in range(Hi):
        for j in range(Wi):
            # Extract the region of interest (ROI) from the padded image
            if num_channels == 1:
                roi = padded_image[i : i + Hk, j : j + Wk]
                out[i, j] = np.sum(roi * kernel)
            else:
                roi = padded_image[i : i + Hk, j : j + Wk, :]
                for c in range(num_channels):
                    out[i, j, c] = np.sum(roi[:, :, c] * kernel)

    return out


def auto_corr(image, template_size):
    """Auto-correlation of an image using a template patch

    Args:
        image: numpy array of shape (Hi, Wi)
        template_size: tuple of integers (Ht, Wt) specifying the size of the template patch

    Returns:
        corr_image: numpy array of shape (Hi, Wi)
        max_similarity: float, maximum similarity value
        max_similarity_location: tuple of integers (y, x), location of maximum similarity
    """
    # Extract the template patch from the image
    Ht, Wt = template_size
    template = image[:Ht, :Wt]

    # Calculate the correlation of the template with every location in the image
    corr_image = corr(image, template)

    # Find the maximum similarity value and its location
    max_similarity = np.max(corr_image)
    max_similarity_location = np.unravel_index(np.argmax(corr_image), corr_image.shape)

    return corr_image, max_similarity, max_similarity_location
