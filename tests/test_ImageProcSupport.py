import numpy as np

# Import functions to be tested
from ImageProcSupport import find_objects_column_gaussian
from ImageProcSupport import fill_holes
from ImageProcSupport import average_column_intensity
from ImageProcSupport import background_correct_with_clipping


#
# Unit tests for find_objects_column_gaussian()
#

# Test case 1: Basic function test with no mask
def test_find_objects_column_gaussian_no_mask():
    image_in = np.array([[10, 20, 30],
                         [1, 2, 3],
                         [2, 3, 4]])

    k = 1.0
    expected_result = np.array([[0, 0, 0],
                                [1, 1, 1],
                                [1, 1, 1]])

    result = find_objects_column_gaussian(image_in, k)
    np.testing.assert_array_equal(result, expected_result)


# Test case 2: Test with mask that masks out object pixels (pixel intensity = 0)
def test_find_objects_column_gaussian_with_mask():
    image_in = np.array([[10, 20, 30],
                         [1, 2, 3],
                         [2, 3, 4]])

    mask = np.array([[1, 1, 0],
                     [1, 0, 1],
                     [0, 1, 1]])
    k = 1.0
    expected_result = np.array([[1, 1, 0],
                                [1, 0, 1],
                                [1, 1, 1]])

    result = find_objects_column_gaussian(image_in, k, mask)
    np.testing.assert_array_equal(result, expected_result)


# Test case 3: Test with a different threshold (k)
def test_find_objects_column_gaussian_custom_threshold():
    image_in = np.array([[10, 20, 30],
                         [1, 2, 3],
                         [2, 3, 3],
                         [5, 6, 3]])
    k = 1.6
    expected_result = np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    result = find_objects_column_gaussian(image_in, k)
    np.testing.assert_array_equal(result, expected_result)


#
# Unit tests for fill_holes()
#

# Test case 1: Test with an image containing holes
def test_fill_holes_with_holes():
    # Create a simple test image with holes
    input_image = np.array([[0, 0, 0, 0, 0],
                            [0, 255, 255, 255, 0],
                            [0, 255, 0, 255, 0],
                            [0, 255, 255, 255, 0],
                            [0, 0, 0, 0, 0]], dtype=np.uint8)

    expected_result = np.array([[0, 0, 0, 0, 0],
                                [0, 255, 255, 255, 0],
                                [0, 255, 255, 255, 0],
                                [0, 255, 255, 255, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)

    result = fill_holes(input_image)
    np.testing.assert_array_equal(result, expected_result)


# Test case 2: Test with an image without holes (no change expected)
def test_fill_holes_without_holes():
    # Create a test image without holes
    input_image = np.array([[0, 0, 0],
                            [0, 255, 0],
                            [0, 0, 0]], dtype=np.uint8)

    expected_result = input_image

    result = fill_holes(input_image)
    np.testing.assert_array_equal(result, expected_result)


# Test case 3: Test with a large object that fills the whole image containing a large hole
def test_fill_holes_large_image_with_holes():
    # Create a larger test image with holes
    input_image = np.array([[255, 255, 255, 255, 255, 255, 255, 255],
                            [255, 0, 0, 0, 0, 0, 0, 255],
                            [255, 0, 0, 0, 0, 0, 0, 255],
                            [255, 0, 0, 0, 0, 0, 0, 255],
                            [255, 0, 0, 0, 0, 0, 0, 255],
                            [255, 0, 0, 0, 0, 0, 0, 255],
                            [255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)

    expected_result = np.array([[255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255],
                                [255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)

    result = fill_holes(input_image)
    np.testing.assert_array_equal(result, expected_result)


#
# Unit tests for average_column_intensity()
#

# Test case 1: Valid input with no masked areas

def test_average_column_intensity_valid_input_no_mask():
    image_gray = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    mask = np.ones_like(image_gray)
    result = average_column_intensity(image_gray, mask)
    expected = [4.0, 5.0, 6.0]
    assert np.allclose(result, expected)


# Test case 2: Valid input with masked areas
def test_average_column_intensity_valid_input_with_mask():
    image_gray = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    mask = np.array([[1, 0, 1],
                     [1, 0, 1],
                     [0, 1, 1]])
    result = average_column_intensity(image_gray, mask)
    expected = [2.5, 8.0, 6.0]
    assert np.allclose(result, expected)


# Test case 3: Unequal dimensions of image and mask.  Expect empty list be returned.
def test_average_column_intensity_incompatible_image_and_mask():
    image_gray = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    mask = np.array([[1, 0, 1],
                     [1, 1, 1]])
    result = average_column_intensity(image_gray, mask)
    expected = []
    assert result == expected


# Test case 4: All pixels are masked out
def test_average_column_intensity_all_pixels_masked():
    image_gray = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    mask = np.zeros_like(image_gray)
    result = average_column_intensity(image_gray, mask)
    expected = [0.0, 0.0, 0.0]
    assert np.allclose(result, expected)


#
# Unit tests for background_correct_with_clipping()
#

# Test case 1: Valid input with no clipping
def test_background_correct_with_clipping_no_clip():
    image = np.array([[100, 150, 200],
                     [50, 100, 150],
                     [0, 50, 100]], dtype=np.uint8)
    correction = np.array([10, 10, 10], dtype=np.int16)

    result = background_correct_with_clipping(image, correction, 0, 255)
    expected = np.array([[110, 160, 210],
                         [60, 110, 160],
                         [10, 60, 110]], dtype=np.uint8)
    assert np.array_equal(result, expected)


# Test case 2: Valid input with clipping
def test_background_correct_with_clipping_clipping():

    image = np.array([[100, 250, 255],
                     [50, 100, 150],
                     [10, 50, 249]], dtype=np.uint8)
    correction = np.array([-50, 10, 10], dtype=np.int16)

    result = background_correct_with_clipping(image, correction, 0, 255)
    expected = np.array([[50, 255, 255],
                         [0, 110, 160],
                         [0, 60, 255]], dtype=np.uint8)
    assert np.array_equal(result, expected)


