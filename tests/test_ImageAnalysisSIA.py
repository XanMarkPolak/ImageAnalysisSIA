
import pytest

from ImageAnalysisSIA import ProcessImagesSIA

# The software can access and load a large set of images that includes all the images acquired during the SIA processing
#   of a single slurry sample.

    # Software verifies presence of images and equal number of NIR and VIS images.

# The sample information will be passed to the application through the command line parameters during the calling of the
#   application by the HMI/Controller software.

# Configuration file specifies the folder location containing the images.

# The software should return an error code if unable to load images.


@pytest.fixture()
def process_images_sia():
    process_images_sia = ProcessImagesSIA('C:\\Users\\markn\\OneDrive\\Xanantec Work\\SIA\\ImageAnalysisSIA\\test_data')
    return process_images_sia


def test_get_file_list_and_verify_correct_files_exist(process_images_sia):
    process_images_sia.get_file_list_and_verify_correct_files_exist()
    assert True
