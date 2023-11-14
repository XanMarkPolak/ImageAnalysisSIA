import os
import pytest
import numpy as np
import csv

from ImageAnalysisSIA import ProcessImagesSIA
from ImageAnalysisSIA import object_properties_to_csv

TEST_TMP_FOLDER = 'C:\\Users\\markn\\OneDrive\\Xanantec Work\\SIA\\ImageAnalysisSIA\\tests\\tmp_tests'


@pytest.fixture()
def process_images_sia():
    cleanup_test_files(TEST_TMP_FOLDER)

    process_images_sia = ProcessImagesSIA(TEST_TMP_FOLDER)

    # create shorter test filenames and define start and end of frame number character
    process_images_sia.start_frame_character = 7
    process_images_sia.end_frame_character = 10
    process_images_sia.first_valid_frame = 1
    process_images_sia.last_valid_frame = -1

    return process_images_sia


# Define a fixture to create a temporary CSV file for testing
@pytest.fixture
def csv_object_file(tmp_path):
    file_path = tmp_path / "test_output.csv"
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def csv_summary_file(tmp_path):
    file_path = tmp_path / "test_summary_output.csv"
    yield file_path
    if file_path.exists():
        file_path.unlink()


# Functions to create and cleanup dummy image files for testing the get_file_list() method.
def create_test_files(base_folder, file_list):
    try:
        # Create the base test folder if it doesn't exist
        os.makedirs(base_folder, exist_ok=True)

        # Create dummy test images with 1-byte content
        for file_name in file_list:
            file_path = os.path.join(base_folder, file_name)
            with open(file_path, 'wb') as file:
                file.write(b'\x00')  # Create a 1-byte file with null content

    except Exception as e:
        raise RuntimeError(f"Error creating test files: {e}")


def cleanup_test_files(base_folder):
    try:
        # Check if the base test folder exists
        if os.path.exists(base_folder):
            # Remove all test files and the base test folder once the test is finished
            for root, dirs, files in os.walk(base_folder, topdown=False):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    os.remove(file_path)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    os.rmdir(dir_path)
            os.rmdir(base_folder)
        else:
            print(f"Test folder '{base_folder}' does not exist. Cleanup skipped.")
    except Exception as e:
        raise RuntimeError(f"Error cleaning up test files: {e}")


#
# Unit tests for testing the get_file_list() method.
#

# Basic Test - Matching VIS and NIR Files:
# Test that the method works correctly when both VIS and NIR images are present in the folder
#   and their frame numbers match.
def test_matching_vis_and_nir_files(process_images_sia):
    # Create test VIS and NIR image files with matching frame numbers in the test folder
    create_test_files(process_images_sia.image_folder, ['LS-VIS-0001.jpg', 'LS-NIR-0001.jpg',
                                                        'LS-VIS-0002.jpg', 'LS-NIR-0002.jpg'])

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-0001.jpg', 'LS-VIS-0002.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-0001.jpg', 'LS-NIR-0002.jpg']


# Test - Missing VIS or NIR Files
# Test that the method detects and handles the case when either VIS or NIR files are missing.
def test_missing_vis_or_nir_files(process_images_sia):
    # Create only VIS files in the test folder
    create_test_files(process_images_sia.image_folder, ['LS-VIS-0001.jpg', 'LS-VIS-0002.jpg'])

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files is None
    assert process_images_sia.nir_image_files is None


# Test - Mismatched Frame Numbers:
# Test that the method handles the case when frame numbers of VIS and NIR images do not match.
def test_mismatched_frame_numbers(process_images_sia):
    # Create test VIS and NIR image files with mismatched frame numbers in the test folder
    create_test_files(process_images_sia.image_folder, ['LS-VIS-0001.jpg', 'LS-NIR-0002.jpg',
                                                        'LS-VIS-0003.jpg', 'LS-NIR-0004.jpg'])

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files is None
    assert process_images_sia.nir_image_files is None


# Test - Empty Folder:
# Test that the method handles the case when the image folder is empty.
def test_empty_folder(process_images_sia):
    # Create and empty folder
    os.makedirs(process_images_sia.image_folder, exist_ok=True)

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files is None
    assert process_images_sia.nir_image_files is None


# Test - Filter Bad Frame Numbers:
# Test that the method filters out non-numeric frame number.
def test_filter_bad_frame_num(process_images_sia):
    # Create test VIS and NIR image files with incorrect frame number format
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-GGG1.jpg', 'LS-NIR-GGG1.jpg', 'LS-VIS-0002.jpg', 'LS-NIR-0002.jpg'])

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-0002.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-0002.jpg']


# Test - Different Number of VIS and NIR Files:
# Test that the method handles cases where there is a different number of VIS and NIR files.
def test_different_number_of_files(process_images_sia):
    # Create different numbers of VIS and NIR image files in the test folder
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-0001.jpg', 'LS-NIR-0001.jpg', 'LS-VIS-0002.jpg'])

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files is None
    assert process_images_sia.nir_image_files is None


# Test for correct frame numbers when within the normal specified range
def test_valid_frame_in_range(process_images_sia):
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-0001.jpg', 'LS-NIR-0001.jpg', 'LS-VIS-0002.jpg', 'LS-NIR-0002.jpg',
                                        'LS-VIS-0003.jpg', 'LS-NIR-0003.jpg', 'LS-VIS-0004.jpg', 'LS-NIR-0004.jpg'])
    process_images_sia.first_valid_frame = 2
    process_images_sia.last_valid_frame = 3

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-0002.jpg', 'LS-VIS-0003.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-0002.jpg', 'LS-NIR-0003.jpg']


# Test for invalid frame numbers out of the specified frame range.
# If invalid frame numbers are outside of range, the program should still return the valid frame numbers.
def test_invalid_frame_numbers_outside_of_range(process_images_sia):
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-GGG1.jpg', 'LS-NIR-GGG1.jpg', 'LS-VIS-0002.jpg', 'LS-NIR-0002.jpg',
                                        'LS-VIS-0003.jpg', 'LS-NIR-0003.jpg', 'LS-VIS-GGG4.jpg', 'LS-NIR-GGG4.jpg'])
    process_images_sia.first_valid_frame = 2
    process_images_sia.last_valid_frame = 3

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-0002.jpg', 'LS-VIS-0003.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-0002.jpg', 'LS-NIR-0003.jpg']


def test_unlimited_frame_number_range(process_images_sia):
    # Test an invalid frame number below the specified range
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-9999.jpg', 'LS-NIR-9999.jpg'])
    process_images_sia.first_valid_frame = -1
    process_images_sia.last_valid_frame = -1

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-9999.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-9999.jpg']


def test_missing_frames_at_start_and_end_of_range(process_images_sia):
    create_test_files(TEST_TMP_FOLDER, ['LS-VIS-0003.jpg', 'LS-NIR-0003.jpg'])
    process_images_sia.first_valid_frame = 1
    process_images_sia.last_valid_frame = 5

    process_images_sia.get_file_list()
    assert process_images_sia.vis_image_files == ['LS-VIS-0003.jpg']
    assert process_images_sia.nir_image_files == ['LS-NIR-0003.jpg']


#
# Unit tests for object_properties_to_csv() function.
#

# Test object_properties_to_csv() for being able to create the target CSV file.
def test_object_properties_to_csv_creation(csv_object_file, csv_summary_file):
    # Create a test binary image
    binary_image = np.array([[0, 1, 0],
                             [0, 1, 0]])

    # Call the function
    object_properties_to_csv(binary_image, csv_object_file, 7.797271, 4000, True, csv_summary_file)

    # Check if the CSV files were created
    assert csv_object_file.exists()
    assert csv_summary_file.exists()


# Test object_properties_to_csv() for one large object to make sure the expected content is written into the CSV file.
def test_object_properties_to_csv_one_object(csv_object_file):
    # Create a test binary image
    binary_image = np.array([[0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0]])

    # Call the function
    object_properties_to_csv(binary_image, csv_object_file, 7.797271, 4000, False)

    # Read the CSV file to verify its contents
    with open(csv_object_file, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Check if the header matches the expected header
        expected_header = ["CentroidX", "CentroidY", "Area(pix)", "Speed(mm/s)", "ObjClass", "MajorAxisLength(pix)",
                           "MinorAxisLength(pix)", "Orientation(rad)", "Solidity", "Eccentricity", "Diameter(um)"]

        # This is the expected result from "measure.regionprops()" function in skimage library
        expected_data_row = ['2.0', '6.0', '39.0', '7.835624753313121', 'BIT', '14.966629547095765',
                             '3.265986323710904', '0.0', '1.0', '0.9759000729485332', '25.465780448267644']

        assert rows[0] == expected_header
        assert rows[1] == expected_data_row


# Test object_properties_to_csv() for 3 objects to make sure the expected content is written into the CSV files.
def test_object_properties_to_csv_and_csv_summary(csv_object_file, csv_summary_file):
    # Create a test binary image
    binary_image = np.array([[0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 255, 255, 255, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 255, 255, 255, 255],
                             [0, 255, 255, 255, 255],
                             [0, 255, 255, 255, 255],
                             [0, 255, 255, 255, 255],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255]])

    # Call the function
    object_properties_to_csv(
        binary_image, csv_object_file, image_scale=1.0, line_scan_rate=1000,
        create_summary_stats=True, summary_csv_file=csv_summary_file
    )

    # Read the CSV file to verify its contents
    with open(csv_object_file, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Check if the header matches the expected header
        expected_header = ["CentroidX", "CentroidY", "Area(pix)", "Speed(mm/s)", "ObjClass", "MajorAxisLength(pix)",
                           "MinorAxisLength(pix)", "Orientation(rad)", "Solidity", "Eccentricity", "Diameter(um)"]

        # This is the expected result from "measure.regionprops()" function in skimage library
        expected_data_row1 = ['2.0', '1.0', '9.0', '1.0886621079036347', 'BIT', '3.265986323710904',
                              '3.265986323710904', '0.7853981633974483', '1.0', '0.0', '3.265986323710904']
        expected_data_row2 = ['2.5', '6.5', '16.0', '1.118033988749895', 'BIT', '4.47213595499958',
                              '4.47213595499958', '0.7853981633974483', '1.0', '0.0', '4.47213595499958']
        expected_data_row3 = ['2.0', '13.0', '25.0', '1.1313708498984762', 'BIT', '5.656854249492381',
                              '5.656854249492381', '0.7853981633974483', '1.0', '0.0', '5.6568542494923815']

        assert rows[0] == expected_header
        assert rows[1] == expected_data_row1
        assert rows[2] == expected_data_row2
        assert rows[3] == expected_data_row3

    # Check if the summary CSV file was created
    assert csv_summary_file.exists()

    # Read the contents of the summary CSV file and perform assertions
    with open(csv_summary_file, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

        assert len(rows) == 12  # Check the number of rows in the summary CSV file is correct

        # Check that number of objects, average diameter, and average speed of the 3 objects is correct
        # Check this for all types of objects: bitumen, air, sand, and unknown.
        assert rows[0] == ["NumOfBitObjects", "AvgDiameter", "AvgSpeed"]
        assert float(rows[1][0]) == 3
        assert float(rows[1][1]) == pytest.approx(4.464992176067621, rel=1e-5)
        assert float(rows[1][2]) == pytest.approx(1.112688982184002, rel=1e-5)
        assert rows[2] == ["NumOfSandObjects", "AvgDiameter", "AvgSpeed"]
        assert float(rows[3][0]) == 0
        assert float(rows[3][1]) == pytest.approx(0, rel=1e-5)
        assert float(rows[3][2]) == pytest.approx(0.0, rel=1e-5)
        assert rows[4] == ["NumOfAirObjects", "AvgDiameter", "AvgSpeed"]
        assert float(rows[5][0]) == 0
        assert float(rows[5][1]) == pytest.approx(0, rel=1e-5)
        assert float(rows[5][2]) == pytest.approx(0.0, rel=1e-5)
        assert rows[6] == ["NumOfUnKnObjects", "AvgDiameter", "AvgSpeed"]
        assert float(rows[7][0]) == 0
        assert float(rows[7][1]) == pytest.approx(0, rel=1e-5)
        assert float(rows[7][2]) == pytest.approx(0.0, rel=1e-5)

        assert rows[8] == ["Diameter(um)", "CPP_by_Diameter)"]
        assert rows[9] == ['3.265986323710904', '24.38217280063487']
        assert rows[10] == ['4.47213595499958', '57.76883791037645']
        assert rows[11] == ['5.6568542494923815', '100.0']


#
# Unit test for load_json_config_file() method.
#

# Test when the config file is successfully loaded
def test_load_json_config_file(monkeypatch):
    process_images_instance = ProcessImagesSIA("")

    process_images_instance.load_json_config_file()

    # Verify that the config_json_loaded attribute is set
    assert process_images_instance.config_json_loaded

    # Verify that the image_scale and line_scan_rate have values other than 0, which is what they are initialized to.
    assert process_images_instance.image_scale > 0
    assert process_images_instance.line_scan_rate > 0
