import numpy as np
from ImageSegmentationSIA import segment_image_set_obj_by_nir
from ImageSegmentationSIA import segment_image_set_by_vis_img


#
# Unit tests for function segment_image_set_obj_by_nir
#

# Simple unit test to check if the function at least returns a valid NumPy arrays and has the correct shape when valid
# images are passed in.
def test_segment_image_set_obj_by_nir_valid_input():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8)  # Example RGB image
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255  # Example NIR image
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert result1.shape == image_nir.shape
    assert isinstance(result2, np.ndarray)
    assert result2.shape == image_nir.shape


# Test if segment_image_set_obj_by_nir() returns None when provided with invalid input (None).
def test_segment_image_set_obj_by_nir_invalid_input():
    result1, result2 = segment_image_set_obj_by_nir(None, None, 2.0, 4.0)
    assert result1 is None
    assert result2 is None


# Test if the function behaves correctly when there are no objects in the images.  It should return a binary,
# but the image should be all 0s.
def test_segment_image_set_obj_by_nir_no_objects():
    image_vis = np.zeros((100, 100, 3), dtype=np.uint8)
    image_nir = np.zeros((100, 100), dtype=np.uint8)
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)  # Check if a NumPy array was returned.
    assert np.all(result1 == 0)  # Check if the returned NumPy array is all 0s, as expected.
    assert isinstance(result2, np.ndarray)  # Check if a NumPy array was returned.
    assert np.all(result2 == 0)  # Check if the returned NumPy array is all 0s, as expected.


# Test if the function correctly removes a small dark object from the binary image.
def test_segment_image_set_obj_by_nir_remove_small_dark_objects():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8)
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255
    # Add a small object in the center
    image_nir[45:50, 45:50] = 0  # This object is only 5 pixels wide, so should be removed.
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0  # Check if the object was removed
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0  # Check if the object was removed


# Test if the function correctly removes a small light object from the binary image.
def test_segment_image_set_obj_by_nir_remove_small_light_objects():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8) * 255
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255
    # Add a small object in the center
    image_nir[45:50, 45:50] = 0  # This object is only 5 pixels wide, so should be removed.
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0  # Check if the object was removed
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0


# Test if the function correctly removes a tall thin dark object from the binary image.
def test_segment_image_set_obj_by_nir_remove_tall_thin_dark_objects():
    image_vis = np.ones((4000, 100, 3), dtype=np.uint8)
    image_nir = np.ones((4000, 100), dtype=np.uint8) * 255
    # Add a tall thin dark object in the center that is 600 pixels high and 20 pixels wide, so that it is 30 times
    # tall as it is wide
    image_nir[300:900, 45:65] = 0
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0  # Ensure tall thin object is removed
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0


# Test if the function correctly removes a tall thin dark object from the binary image.
def test_segment_image_set_obj_by_nir_remove_tall_thin_light_objects():
    image_vis = np.ones((4000, 100, 3), dtype=np.uint8) * 255
    image_nir = np.ones((4000, 100), dtype=np.uint8) * 255
    # Add a tall thin dark object in the center that is 600 pixels high and 20 pixels wide, so that it is 30 times
    # tall as it is wide
    image_nir[300:900, 45:65] = 0
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0  # Ensure tall thin object is removed


# Test if the function correctly finds an object seen in NIR image that matches colour of bitumen.
def test_segment_image_set_obj_by_nir_correctly_segment_bitumen_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)  # Intensity of 1 is dark enough
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) / 255 == 20 * 20  # Ensure that a 20x20 bitumen object is identified in resulting image
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0  # Other resulting image should be empty.


# Test if the function correctly finds an object seen in NIR image that matches colour of air/sand.
def test_segment_image_set_obj_by_nir_correctly_segment_light_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    image_vis[300:320, 45:65, :] = 250  # Intensity of 250 is light enough

    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0  # Bitumen resulting image should be empty.
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) / 255 == 20 * 20  # Ensure that a 20x20 light object is identified in resulting image


# Test if the function correctly finds multiple objects seen in NIR image and identifies them as bitumen or other.
def test_segment_image_set_obj_by_nir_correctly_segment__multiple__objects():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)  # Intensity of 1 is dark
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Create 20 objects with 20x20 dimensions: 10 dark and 10 light objects
    for r in range(20, 1000, 200):
        # Add a valid size object
        image_nir[r:r + 20, 0:20] = 0  # dark object

        image_nir[r:r + 20, 25:45] = 0
        image_vis[r:r + 20, 25:45, :] = 200     # light object

        image_nir[r:r + 20, 50:70] = 0  # dark object

        image_nir[r:r + 20, 75:95] = 0
        image_vis[r:r + 20, 75:95, :] = 200     # light object

    result1, result2 = segment_image_set_obj_by_nir(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) / 255 == 10 * 20 * 20  # Ensure that 10 20x20 bitumen objects are identified.
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) / 255 == 10 * 20 * 20  # Ensure that 10 20x20 bitumen objects are identified.


#
# Unit tests for function segment_image_set_by_vis_img
#

# Simple unit test to check if the function at least returns a valid NumPy arrays and has the correct shape when valid
# images are passed in.
def test_segment_image_set_by_vis_img_valid_input():
    image_vis = np.ones((100, 100, 3), dtype=np.uint8)  # Example RGB image
    image_nir = np.ones((100, 100), dtype=np.uint8) * 255  # Example NIR image
    result1, result2 = segment_image_set_by_vis_img(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)
    assert result1.shape == image_nir.shape
    assert isinstance(result2, np.ndarray)
    assert result2.shape == image_nir.shape


# Test if segment_image_set_by_vis_img() returns None when provided with invalid input (None).
def test_segment_image_set_by_vis_img_invalid_input():
    result1, result2 = segment_image_set_by_vis_img(None, None, 2.0, 4.0)
    assert result1 is None
    assert result2 is None


# Test if the function behaves correctly when there are no objects in the images.  It should return a binary,
# but the image should be all 0s.
def test_segment_image_set_by_vis_img_no_objects():
    image_vis = np.zeros((100, 100, 3), dtype=np.uint8)
    image_nir = np.zeros((100, 100), dtype=np.uint8)
    result1, result2 = segment_image_set_by_vis_img(image_vis, image_nir, 2.0, 4.0)
    assert isinstance(result1, np.ndarray)  # Check if a NumPy array was returned.
    assert np.all(result1 == 0)  # Check if the returned NumPy array is all 0s, as expected.
    assert isinstance(result2, np.ndarray)  # Check if a NumPy array was returned.
    assert np.all(result2 == 0)  # Check if the returned NumPy array is all 0s, as expected.


# Test if the function correctly finds an object seen in NIR image that matches colour of bitumen.
def test_segment_image_set_by_vis_img_correctly_segment_bitumen_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8) * 120  # Intensity of 120 is background
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    image_vis[300:320, 45:65, :] = 0  # A very dark object (bitumen)

    result1, result2 = segment_image_set_by_vis_img(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) / 255 == 20 * 20  # Ensure that a 20x20 bitumen object is identified in resulting image
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) == 0  # Other resulting image should be empty.


# Test if the function correctly finds an object seen in NIR image that matches colour of air/sand.
def test_segment_image_set_by_vis_img_correctly_segment_light_object():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8) * 120  # Intensity of 120 is background
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Add a valid size object
    image_nir[300:320, 45:65] = 0
    image_vis[300:320, 45:65, :] = 250  # Intensity of 250 is light enough

    result1, result2 = segment_image_set_by_vis_img(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) == 0  # Bitumen resulting image should be empty.
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) / 255 == 20 * 20  # Ensure that a 20x20 light object is identified in resulting image


# Test if the function correctly finds multiple objects seen in NIR image and identifies them as bitumen or other.
def test_segment_image_set_by_vis_img_correctly_segment__multiple__objects():
    image_vis = np.ones((1000, 100, 3), dtype=np.uint8)  * 120  # Intensity of 120 is background
    image_nir = np.ones((1000, 100), dtype=np.uint8) * 255

    # Create 20 objects with 20x20 dimensions: 10 dark and 10 light objects
    for r in range(20, 1000, 200):
        # Add a valid size object
        image_nir[r:r + 20, 0:20] = 0  # object
        image_vis[r:r + 20, 0:20] = 0  # dark object

        image_nir[r:r + 20, 25:45] = 0  # object
        image_vis[r:r + 20, 25:45, :] = 220     # light object

        image_nir[r:r + 20, 50:70] = 0  # object
        image_vis[r:r + 20, 50:70] = 0  # dark object

        image_nir[r:r + 20, 75:95] = 0  # object
        image_vis[r:r + 20, 75:95, :] = 200     # light object

    result1, result2 = segment_image_set_by_vis_img(image_vis, image_nir, 2.0, 4.0)

    assert isinstance(result1, np.ndarray)
    assert np.sum(result1) / 255 == 10 * 20 * 20  # Ensure that 10 20x20 bitumen objects are identified.
    assert isinstance(result2, np.ndarray)
    assert np.sum(result2) / 255 == 10 * 20 * 20  # Ensure that 10 20x20 bitumen objects are identified.
