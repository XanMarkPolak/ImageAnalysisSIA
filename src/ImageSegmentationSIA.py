#
# ImageSegmentationSIA.py
#
# This file contains the main image processing functions.
#
# Written by: Mark Polak
#
# Date Created: 2023-10-31
# Last Modified: 2023-11-14
#

import numpy as np
import cv2

import ImageProcSupport as ipsSIA

MAX_HEIGHT_TO_WIDTH_RATIO = 25
MIN_OBJECT_WIDTH = 15


def segment_image_set_obj_by_nir(image_vis, image_nir):
    """
     Segment an image set by detecting objects using near-infrared (NIR) image and then using the RGB image to
     determine if each detected object is bitumen.

     Parameters:
     - image_vis (numpy.ndarray): RGB color image.
     - image_nir (numpy.ndarray): Corresponding NIR image.

     Returns:
     - numpy.ndarray: Binary image with detected dark objects.
     - numpy.ndarray: Binary image with detected light objects.

     Algorithm Steps:
     1. Apply a two-pass Gaussian background detection model to the NIR image to create an object mask.
        The object mask is used to get a more accurate model of the background by analyzing only background
        pixels (objects are masked out).  This object mask is used to correct the lighting variations in the RGB image
        across columns and to identify each object for analysis.
     2. Convert the RGB image section to a single-channel grayscale image.
     3. Calculate the average intensity of background pixels for each column in the RGB image.
     4. Correct the RGB image intensity to make it uniform across columns.
     5. Create a binary mask for detected objects using the second-pass Gaussian model.
     6. Fill holes in the object mask to create a solid mask of detected objects.
     7. Perform morphological operations to eliminate small objects and artifacts.
     8. Label the connected components in the binary image.
     9. Analyze each labeled object to filter out small artifacts and tall-thin artifacts.
     10. According to intensity of object pixels in RGB image, classify the image as bitumen or other.
     10. Create a new binary image for bitumen objects.
     11. Create a new binary image for non-bitumen objects.
     11. Return the two final binary images with detected objects.

     Note:
     - The function utilizes custom functions from the ImageProcSupport module for background detection and corrections.

     Example Usage:
     ```
     result_dark_binary_image, result_other_binary_image = segment_image_set_obj_by_nir(rgb_image, nir_image)
     ```

     """

    # Check if the image was loaded successfully
    if image_vis is None or image_nir is None:
        return None, None

    # Get the dimensions of the image
    height, width, channel = image_vis.shape

    #
    # Get a background model through two passes of a background detection model that is based on Gaussian method on
    # each individual column.
    #

    # Apply custom thresholding using Gaussian background model
    # This is the first pass, so variance in the Gaussian model will be high because both background and object
    # pixels are used in calculation.
    k = 2.0  # Typically k would be 2.5, but want to make sure we don't miss pixels of objects.
    mask_level1 = ipsSIA.find_objects_column_gaussian(image_nir, k, None)

    # Apply custom thresholding using Gaussian background model for a second pass.  The object mask generated
    # in the first pass is used to mask our object pixels, so mostly background pixels should be included.
    # Because only background pixels are included, the variance of pixel intensity will be low, and as a
    # result the Z score calculated for each pixel in each column will be higher and so a higher k value is needed.
    k = 4.0  # Need to increase k because variance is now reduced
    mask_level2 = ipsSIA.find_objects_column_gaussian(image_nir, k, mask_level1)

    #
    # Use the object mask created from the NIR image to calculate the average intensity of background pixels for
    # each column in the RGB image.  This average intensity value per column is used to correct the difference in
    # lighting across the RGB image.
    #

    # Convert RGB section to single-channel
    gray_vis_img = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)

    # Find the average column intensities
    average_intensity_per_column = ipsSIA.average_column_intensity(gray_vis_img, mask_level2)

    # Calculate the column pixel correction to make the image intensity uniform across columns
    average_img_intensity = np.mean(average_intensity_per_column)
    column_correction = average_img_intensity - average_intensity_per_column

    #
    # Correct the image intensity to be uniform across columns.
    #
    corrected_vis_img = ipsSIA.background_correct_with_clipping(gray_vis_img, column_correction, 0, 255)

    # Convert any 0s in the mask to 255 (object) and any 1s in the mask to 0 (background).
    # Copy resulting mask into an image padded with 0s all around so that flood fill in fill_holes() does not
    # flood the entire image if there is an object in its seed location (0,0).
    objects = np.zeros((height + 10, width + 10), dtype=np.uint8)
    objects[5:(height + 5), 5:(width + 5)] = 255 * (1 - mask_level2)

    # Fill holes in mask because it will be used as object list
    objects_filled = ipsSIA.fill_holes(objects)

    # Perform morphological open to eliminate small objects.
    kernel = np.ones((3, 3), np.uint8)
    closed_img = cv2.morphologyEx(objects_filled[5:(height + 5), 5:(width + 5)], cv2.MORPH_OPEN,
                                  kernel)  # Remove tiny fragments

    # Label the objects in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_img, connectivity=8)

    #
    # Go through each labeled object and decide whether it is likely to be valid bitumen droplet or air/sand object.
    # If not, then have it removed.
    #

    # Initialize a set to keep track of dark (bitumen) and light (air or sand) objects
    dark_objects = set()
    light_objects = set()

    avg_bitumen_intensity = average_img_intensity + 10

    # Iterate through labeled objects
    for label in range(1, num_labels):
        obj_width = stats[label, cv2.CC_STAT_WIDTH]

        # Only keep objects of width greater than MIN_OBJECT_WIDTH.  Ignore small artifacts (object of width less
        # than 15 pixels)
        # Also only keep object where the height of the object is less than width times MAX_HEIGHT_TO_WIDTH_RATIO.
        # Ignore extremely tall thin object (dirt on lens).
        if obj_width > MIN_OBJECT_WIDTH and stats[label, cv2.CC_STAT_HEIGHT] < MAX_HEIGHT_TO_WIDTH_RATIO * obj_width:
            # Extract the region corresponding to the labeled object
            object_region = (labels == label)

            # Find average pixel value in the corresponding region of the corrected RGB image (corrected_vis_img)
            # If the object is dark enough to be bitumen, add it to the set of dark_objects.
            # Otherwise, add it to the set of light_objects.
            avg_pixel_value = np.average(corrected_vis_img[object_region])

            if avg_pixel_value < avg_bitumen_intensity:
                # Object is dark enough to be bitumen.   Add it.
                dark_objects.add(label)
            else:
                # Otherwise it's not dark enough to be bitumen, so add it to set of light objects.
                light_objects.add(label)

    #
    # Create a new binary image without the objects marked for removal.  This will be the final binary image
    # that is returned.
    #
    new_dark_binary_image = np.zeros_like(closed_img)
    new_light_binary_image = np.zeros_like(closed_img)
    for label in range(1, num_labels):
        if label in dark_objects:
            new_dark_binary_image[labels == label] = 255  # Set retained dark objects to 255 (white)
        elif label in light_objects:
            new_light_binary_image[labels == label] = 255  # Set retained light objects to 255 (white)

    return new_dark_binary_image, new_light_binary_image


def segment_image_set_by_vis_img(image_vis, image_nir):
    """
     Segment an image set by detecting objects using visual RGB (VIS) image and then using the near infrared (NIR)
     image to correct the segmentation.

     Parameters:
    - image_vis (numpy.ndarray): RGB color image.
    - image_nir (numpy.ndarray): Corresponding 8-bit NIR image.

     Returns:
     - numpy.ndarray: Binary image with detected dark objects.
     - numpy.ndarray: Binary image with detected light objects.

     Algorithm Steps:
    1. Input Validation:
        Check if the input images (image_vis and image_nir) are loaded successfully. If not, return None for both
        binary images.
    2. Background Separation:
        Use a Gaussian background model on each column of the NIR image for background separation.
        Apply custom thresholding using a Gaussian background model in two passes (k_level1 and k_level2).
    3. Calculate Column-wise Statistics for Background:
        For each column, calculate the average background intensity in the RGB image separately for red, green, and
        blue channels, using only background pixels defined by the object mask created from the NIR image.
    4. Generate Difference Images:
        Find the difference in each column and channel between the average background and pixel values in the RGB image.
        For dark objects, use the maximum difference to identify these objects.
        For light objects, use the minimum difference because they are distinct.
    5. Thresholding:
        Threshold the difference images obtained in the previous step using a low threshold (thresh_level) to ensure
        identification of any pixels belonging to an object.
        Apply a mask created from the background model to eliminate artifacts left over from the threshold operation.
    6.  Fill Holes:
        Fill holes in the binary images to complete objects.
    7. Morphological Operations:
        Perform morphological "open" operation to eliminate small objects.
    8. Return Results:
        Return the binary images representing dark and light objects.

     Note:
     - The function utilizes custom functions from the ImageProcSupport module for background detection and corrections.

     Example Usage:
     ```
     result_dark_binary_image, result_other_binary_image = segment_image_set_by_vis_img(rgb_image, nir_image)
     ```

     """
    # Check if the image was loaded successfully
    if image_vis is None or image_nir is None:
        return None, None

    # Get the dimensions of the image
    height, width, channel = image_vis.shape

    #
    # Get a background/foreground separation of the NIR image using background detection model that is based on
    # Gaussian method on each individual column.
    #

    # Apply custom thresholding using Gaussian background model
    # This is the first pass, so variance in the Gaussian model will be high because both background and object
    # pixels are used in calculation.
    k = 2.0  # Typically k would be 2.5, but want to make sure we don't miss pixels of objects.
    mask_level1 = ipsSIA.find_objects_column_gaussian(image_nir, k, None)

    # Apply custom thresholding using Gaussian background model for a second pass.  The object mask generated
    # in the first pass is used to mask our object pixels, so mostly background pixels should be included.
    # Because only background pixels are included, the variance of pixel intensity will be low, and as a
    # result the Z score calculated for each pixel in each column will be higher and so a higher k value is needed.
    k = 4.0  # Need to increase k because variance is now reduced
    mask_level2 = ipsSIA.find_objects_column_gaussian(image_nir, k, mask_level1)

    # Initialize lists to store column-wise statistics for each color channel
    avg_intensity_red_per_column = []
    avg_intensity_green_per_column = []
    avg_intensity_blue_per_column = []

    #
    # Calculate the average background intensity in the RGB image for each column in each color channel separately.
    # Use only background pixels, as defined by the object mask created from the NIR image.
    #
    for col in range(width):
        column_data_blue = image_vis[:, col, 0].copy()
        column_data_blue *= mask_level2[:, col]
        column_data_green = image_vis[:, col, 1].copy()
        column_data_green *= mask_level2[:, col]
        column_data_red = image_vis[:, col, 2].copy()  # Make a copy of the pixel values for the column
        column_data_red *= mask_level2[:, col]  # Mask out any objects so background is created from background only

        sum_intensity_background = np.sum(column_data_red)
        num_pix_background = np.sum(mask_level2[:, col])
        average_intensity = sum_intensity_background / num_pix_background
        avg_intensity_red_per_column.append(average_intensity)

        sum_intensity_background = np.sum(column_data_green)
        average_intensity = sum_intensity_background / num_pix_background
        avg_intensity_green_per_column.append(average_intensity)

        sum_intensity_background = np.sum(column_data_blue)
        average_intensity = sum_intensity_background / num_pix_background
        avg_intensity_blue_per_column.append(average_intensity)

    # Use an image padded with 0s all around so that flood fill in fill_holes(), that is applied later,
    # does not flood the entire image if there is an object in its seed location (0,0).
    foreground_image_dark_obj = np.zeros((height + 10, width + 10), dtype=np.uint8)
    foreground_image_light_obj = np.zeros((height + 10, width + 10), dtype=np.uint8)

    #
    # Find the difference in each column and channel between the average background and pixel in the RGB image.
    # For dark objects, using the maximum difference is needed to pick out these objets.
    # For light objects, using the minimum difference is better because they are so distinct.  The air and sand
    # seem to be almost completely white and have fairly high intensity in all 3 channels.
    #
    for col in range(width):
        # For dark objects, which have low pixel intensity values, take the average column background intensity and
        # subtract the pixels in the RGB (visual) image.  Do this for each color channel.
        blue_dif = avg_intensity_blue_per_column[col] - image_vis[:, col, 0]
        green_dif = avg_intensity_green_per_column[col] - image_vis[:, col, 1]
        red_dif = avg_intensity_red_per_column[col] - image_vis[:, col, 2]

        # For light objects, for each color channel, take the pixels in the RGB (visual) image and subtract
        # average column background intensity.
        blue_dif_light = image_vis[:, col, 1] - avg_intensity_blue_per_column[col]
        green_dif_light = image_vis[:, col, 1] - avg_intensity_green_per_column[col]
        red_dif_light = image_vis[:, col, 2] - avg_intensity_red_per_column[col]

        # For dark objects, take the maximum difference in all the channels.  In some images, the average pixel
        # intensity of all 3 channels of bitumen droplets is not any darker than the background, so we look at
        # any one channel that might differentiate the background from dark bitumen droplets.
        max_diff = np.maximum.reduce([red_dif, green_dif, blue_dif])  # Find max diff.

        # For light objects, take the minimum difference in all the channels.  The light objects seem to be white,
        # with all channels having high pixel intensity.  For light objects, taking the minimum difference works better.
        min_diff_light = np.minimum.reduce([red_dif_light, green_dif_light, blue_dif_light])  # Find min diff.

        # Clip any negative values at 0.
        max_diff[max_diff < 0] = 0
        min_diff_light[min_diff_light < 0] = 0

        # Coppy the pixel difference arrays into ones padded with 0s, so that flood fill does not flood the
        # entire image if there is an object in its seed location.
        foreground_image_dark_obj[5:(height + 5), col + 5] = max_diff
        foreground_image_light_obj[5:(height + 5), col + 5] = min_diff_light

    # Threshold the differences found in the previous step.  We use a low threshold to make sure we identify
    # any pixels belonging to an object.  Any artifacts that are a consequence of the low threshold
    #  are removed at later stage.
    thresh_level = 5
    _, binary_image_dark_obj = cv2.threshold(foreground_image_dark_obj, thresh_level, 255, cv2.THRESH_BINARY)
    _, binary_image_light_obj = cv2.threshold(foreground_image_light_obj, thresh_level, 255, cv2.THRESH_BINARY)

    # Create a mask that keeps objects and removes background, instead of one that keeps the background and removes
    # objects, as we used previously.
    obj_img = 255 * (1 - mask_level2)

    # Take an intersection between the threshold image and object mask.  This is used to eliminate any
    # artifacts left over from the threshold operation.
    binary_image_dark_obj[5:(height + 5), 5:(width + 5)] = binary_image_dark_obj[5:(height + 5),
                                                                5:(width + 5)] & obj_img
    binary_image_light_obj[5:(height + 5), 5:(width + 5)] = binary_image_light_obj[5:(height + 5),
                                                                5:(width + 5)] & obj_img

    # Fill holes in objects.
    binary_image_dark_obj = ipsSIA.fill_holes(binary_image_dark_obj)
    binary_image_neg_light_obj = ipsSIA.fill_holes(binary_image_light_obj)

    # Perform morphological "open" operation to eliminate small objects.
    kernel = np.ones((3, 3), np.uint8)
    new_dark_binary_image = cv2.morphologyEx(binary_image_dark_obj[5:(height + 5), 5:(width + 5)], cv2.MORPH_OPEN,
                                             kernel)  # Remove tiny dark fragments
    new_light_binary_image = cv2.morphologyEx(binary_image_neg_light_obj[5:(height + 5), 5:(width + 5)], cv2.MORPH_OPEN,
                                              kernel)  # Remove tiny light fragments

    return new_dark_binary_image, new_light_binary_image
