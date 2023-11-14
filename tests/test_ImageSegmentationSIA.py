import numpy as np
import cv2
import pytest
from ImageSegmentationSIA import segment_image_set_obj_by_nir  # Replace 'your_module' with the actual module name


# Simple unit test to check if the function at least returns a valid NumPy array and has the correct shape when valid
# images are passed in.
def test_segment_image_set_obj_by_nir_valid_input():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8)  # Example RGB image
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255  # Example NIR image
    result = segment_image_set_obj_by_nir(image_vis, image_nir)
    assert isinstance(result, np.ndarray)
    assert result.shape == image_nir.shape


# Test if segment_image_set_obj_by_nir() returns None when provided with invalid input (None).
def test_segment_image_set_obj_by_nir_invalid_input():
    result = segment_image_set_obj_by_nir(None, None)
    assert result is None


# Test if the function behaves correctly when there are no objects in the images.  It should return a binary,
# but the image should be all 0s.
def test_segment_image_set_obj_by_nir_no_objects():
    image_vis = np.zeros((100, 100, 3), dtype=np.uint8)
    image_nir = np.zeros((100, 100), dtype=np.uint8)
    result = segment_image_set_obj_by_nir(image_vis, image_nir)
    assert isinstance(result, np.ndarray)  # Check if a NumPy array was returned.
    assert np.all(result == 0)  # Check if the returned NumPy array is all 0s, as expected.


# Test if the function correctly removes a small object from the binary image.
def test_segment_image_set_obj_by_nir_remove_small_objects():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8)
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255
    # Add a small object in the center
    image_nir[45:50, 45:50] = 0  # This object is only 5 pixels wide, so should be removed.
    result = segment_image_set_obj_by_nir(image_vis, image_nir)
    assert isinstance(result, np.ndarray)
    assert np.sum(result) == 0  # Check if the object was removed


# Test if the function correctly removes a tall thin object from the binary image.
def test_segment_image_set_obj_by_nir_remove_tall_thin_objects():
    image_vis = np.ones((4000, 100, 3), dtype=np.uint8)
    image_nir = np.ones((4000, 100), dtype=np.uint8) * 255
    # Add a tall thin object in the center that is 600 pixels high and 20 pixels wide, so that it is 30 times
    # tall as it is wide
    image_nir[300:900, 45:65] = 0
    result = segment_image_set_obj_by_nir(image_vis, image_nir)
    assert isinstance(result, np.ndarray)
    assert np.sum(result) == 0  # Ensure tall thin object is removed


# Test if the function correctly finds an object seen in NIR image that matches colour of bitumen.
def test_segment_image_set_obj_by_nir_correctly_segment_bitumen_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)  # Intensity of 1 is dark enough
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    result = segment_image_set_obj_by_nir(image_vis, image_nir)

    assert isinstance(result, np.ndarray)
    assert np.sum(result) / 255 == 20 * 20  # Ensure that a 20x20 bitumen object is identified in resulting image


# Test if the function correctly finds multiple objects seen in NIR image that matches colour of bitumen.
def test_segment_image_set_obj_by_nir_correctly_segment__multiple_bitumen_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)  # Intensity of 1 is dark enough
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Create 20 objects with 20x20 dimensions.
    for r in range(20, 1000, 200):
        # Add a valid size object
        image_nir[r:r + 20, 0:20] = 0
        image_nir[r:r + 20, 25:45] = 0
        image_nir[r:r + 20, 50:70] = 0
        image_nir[r:r + 20, 75:95] = 0

    result = segment_image_set_obj_by_nir(image_vis, image_nir)

    assert isinstance(result, np.ndarray)
    assert np.sum(result) / 255 == 20 * 20 * 20  # Ensure that 16 20x20 bitumen objects are identified.


# Test if the function correctly ignores an object seen in NIR image that is too bright to be bitumen.
def test_segment_image_set_obj_by_nir_correctly_ignore_non_bitumen_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)*100  # Background intensity is 100 in each channel
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    image_vis[300:320, 45:65, :] = 150   # The object intensity is 150, which is significantly brighter than background

    result = segment_image_set_obj_by_nir(image_vis, image_nir)

    assert isinstance(result, np.ndarray)
    assert np.sum(result) == 0   # Ensure that the bright object is not segmented as bitumen
