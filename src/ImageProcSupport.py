#
# ImageProcSupport.py
#
# This file contains image processing support functions used in the SIA image processing methods.
#
# Written by: Mark Polak
#
# Date Created: 2023-10-30
# Last Modified: 2023-10-30
#

import numpy as np
import cv2


def find_objects_column_gaussian(image_in, k, mask=None):
    """
    Find objects using a Gaussian model.

    Description:
    Fit a Gaussian probability density function (pdf) to LineScan rows to create a background model.
    The pdf of every pixel in the LineScan camera scan (column in the image) is characterized by its mean and variance.
    The calculated mean and variance are used to transform each pixel into a standard scale with a mean of 0 and
    a standard deviation of 1. This standardized score allows us to assess how many standard deviations an
    individual pixel intensity is from the mean. A threshold, k, is used to separate the background and object.

    Usage:
    This function is used to create an initial object mask from a NIR (Near-Infrared) image. The resulting image
    contains values of 0 for the object and 1 for the background. This image can be used as a new object mask.

    Parameters:

    - image_in (numpy.ndarray): An image in NumPy array format loaded with OpenCV (cv2.imread) with dtype=uint8.
      The input image to be segmented into background (0) and objects (1).
    - k (float): The number of standard deviations a pixel must differ from the mean to be classified as an object.
    - mask (numpy.ndarray, optional): An optional parameter. If provided, this parameter is an image with 0 for objects
      and 1 for background. This mask image is used to "mask out" object pixels in the original image.

    Returns:

    - numpy.ndarray: A segmented binary image, where each pixel is identified as an object (0) or background (1).

    Example:
    >>> imgNIR = cv2.imread("LS-NIR-00001-0008-99999.png", cv2.IMREAD_GRAYSCALE)
    >>> result = find_objects_column_gaussian(imgNIR, k)
    # 'result' now contains the segmented binary image.
    """

    height, width = image_in.shape

    # Initialize lists to store column-wise statistics (mean and var)
    average_intensity_per_column = []
    variance_per_column = []

    # Iterate through each column in image_in
    for col in range(width):
        column_data = image_in[:, col]  # Get the pixel values for the column

        if mask is not None:
            # Filter out pixel values where the mask is 0
            valid_pixels = column_data[mask[:, col] == 1]
            average_intensity = np.mean(valid_pixels)  # Calculate average intensity
            variance = np.var(valid_pixels)  # Calculate variance in pixel intensity
        else:
            # If mask is not passed in, calculate mean and variance for all pixels in the column
            average_intensity = np.mean(column_data)  # Calculate average intensity
            variance = np.var(column_data)  # Calculate variance.  Note that np.var calculates population variance.

        average_intensity_per_column.append(average_intensity)
        variance_per_column.append(variance)

    # Create a new image with the same dimensions as the original image
    new_image_mask = np.copy(image_in)

    # Subtract the column-wise average from each pixel in the new image
    for col in range(width):
        avg = average_intensity_per_column[col]
        std = (variance_per_column[col]) ** 0.5

        # Check if image pixel minus column average, divided by column standard deviation (Z score) is greater
        #   than k.   If so, it means this pixel is an object and value 0 is assigned to this pixel, else a
        #   value of 1 is assigned.  This will create a new object mask.
        new_image_mask[:, col] = np.where(np.abs((new_image_mask[:, col] - avg) / np.where(std > 1, std, 1)) > k, 0,
                                          1).astype(np.uint8)

    return new_image_mask


def fill_holes(input_image):
    """
    Fill holes in objects within an image using the cv2.floodFill method.

    This function takes an input image with objects that may contain holes and fills those holes.
    It begins by creating a flood-filled copy of the input image, then inverts it to produce a mask
    for the holes. Finally, it combines the input image with the inverted flood-filled image to fill
    the holes in the objects.

    Parameters:
    input_image (numpy.ndarray):
        The input image with objects that may contain holes (object = 255, background = 0).

    Returns:
    numpy.ndarray:
        An image with holes in objects filled.

    Example:
    >>> input_img = cv2.imread("image_with_holes.png", cv2.IMREAD_GRAYSCALE)
    >>> filled_img = fill_holes(input_img)
    # 'filled_img' now contains the input image with holes in objects filled.
    """

    # Make a copy of the input image and make sure the data type is uint8 for floodFill
    im_flood_fill = input_image.copy().astype("uint8")

    # Get the height and width of the input image
    h, w = input_image.shape[:2]

    # Create a mask for floodFill
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Perform floodFill to fill the objects, starting at seed coordinate (0,0)
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)

    # Invert the flood-filled image to create a mask for the holes
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)

    # Combine the input image with the inverted flood-filled image to fill the holes
    img_filled = input_image | im_flood_fill_inv

    return img_filled


def average_column_intensity(image_gray, mask):
    """
    Calculate the average intensity for each column of an image while excluding masked areas.

    Parameters:
        image_gray (numpy.ndarray): Grayscale image.
        mask (numpy.ndarray): Binary mask where 1 indicates areas to include, 0 to exclude.

    Returns:
        list: List of average intensities per column.
    """

    # Make sure the image and mask are same shape
    if image_gray.shape != mask.shape:
        return []

    masked_image = image_gray * mask  # Apply the mask directly to the image

    # Calculate the sum of pixel intensities for each column
    sum_intensities = np.sum(masked_image, axis=0)

    # Count the number of pixels in each column that are not masked out
    num_pixels = np.sum(mask, axis=0)

    # Avoid division by zero by setting num_pixels to 1 where it's zero
    num_pixels[num_pixels == 0] = 1

    # Calculate average intensity for each column
    average_intensity_per_column = sum_intensities / num_pixels

    return average_intensity_per_column


def background_correct_with_clipping(image, correction, clip_min, clip_max):
    """
    Adjust the intensity of each column in an image with a correction factor, while clipping values within a specified
    range.

    This function takes an input image and applies a correction to it along each column. The resulting values are
    clipped within the range defined by `clip_min` and `clip_max`. If any values go below `clip_min`, they
    are set to `clip_min`, and if any values exceed `clip_max`, they are set to `clip_max`.
    The corrected image is then cast back to uint8 data type.
    It is assumed that image and correction have equal width.

    Parameters:
    image (numpy.ndarray):
        The input image, represented as a NumPy array, with pixel intensities. It should be of data type uint8.

    correction (numpy.ndarray):
        The correction factor to be added to each column in the image. It should be of data type int16 to handle
        potential overflow issues.

    clip_min (int):
        The minimum value to which pixel intensities are clipped. It defines the lower bound of pixel values in
        the corrected image.

    clip_max (int):
        The maximum value to which pixel intensities are clipped. It defines the upper bound of pixel values in
        the corrected image.

    Returns:
    numpy.ndarray:
        An image with adjusted column-wise intensities, where values are clipped within the specified range, and
        the data type is uint8.

    """

    # Convert image to type int16 and add the column correction to it.
    subtracted_image = image.astype(np.int16) + correction.astype(np.int16)

    # If any pixel value is above clip_max or below clip_min, then clip it.
    subtracted_image = np.maximum(subtracted_image, clip_min)
    subtracted_image = np.minimum(subtracted_image, clip_max)

    # Convert back to uint8 and return the corrected image.
    return subtracted_image.astype(np.uint8)
