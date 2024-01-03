#
# SysConfig.py
#
# This file contains configuration used in the SIA image processing methods.  It defines constants imported and used
# in various modules of the software.
# These constants should not need to be changed by the operator or control software that calls SIA image analysis.
#
# Written by: Mark Polak
#
# Date Created: 2023-11-11
# Last Modified: 2023-12-29
#


# Defined absolute paths for:
# 1) JSON configuration file that must be loaded in.  A valid JSON configuration file for SIA must exist in the
#    specified location.
# 2) SIA error log file.  If file does not exist, it will be created, so just the path must be valid and have write
#    access.
#
# *** These paths will have to be modified for each machine that the SIA image analysis software is installed on.
# *** Make sure the paths and filenames are correct before running the software.

OPERATOR_CONFIG_FILE_JSON = "C:\\Users\\markn\\OneDrive\\Xanantec Work\\SIA\\ImageAnalysisSIA\\src\\OperatorConfigSIA.json"
ERROR_LOG_FILE = "C:\\Users\\markn\\OneDrive\\Xanantec Work\\SIA\\ImageAnalysisSIA\\ErrorsSIA.log"
OUTPUT_FOLDER = "C:\\Users\\markn\\OneDrive\\Xanantec Work\\SIA\\ImageAnalysisSIA\\Output"

# Defined SIA system constants.
FIRST_IMAGE_NAME_FRAME_CHARACTER = 13  # first character in image filename that specifies frame number.
LAST_IMAGE_NAME_FRAME_CHARACTER = 16  # last character in image filename that specifies frame number.
FIRST_IMAGE_NAME_SAMPLE_CHARACTER = 7  # first character in image filename that specifies frame number.
LAST_IMAGE_NAME_SAMPLE_CHARACTER = 11  # last character in image filename that specifies frame number.

# Define frames and region of interest in image.
# These parameters should only be changed during system calibration when SIA is first started.
FIRST_FRAME_NUM = -1  # first frame number to process. -1 means whatever the first frame number is in the image set.
LAST_FRAME_NUM = -1  # last frame number to process.  -1 means whatever the last frame number is in the image set.
BAD_EDGE_LEFT = 350
BAD_EDGE_RIGHT = 350

# String constants
CSV_FILE_PREFIX = "LSCAN-Res-"  # File name prefix for CSV files (object file and summary file).
LS_NIR_PREFIX = "LS-NIR-"   # File name prefix for LineScan NIR image files
LS_VIS_PREFIX = "LS-VIS-"   # File name prefix for LineScan visual image files

# Option to downscale the images to fraction of the original size in order to improve runtime performance.
# When downscaling, results in CSV files are automatically adjusted to represent full size images.
# DOWNSCALE_FACTOR of 1 would mean no downscaling.   DOWNSCALE_FACTOR of 0.5 means width and height are reduced by half.
# Whether DOWNSCALE_FACTOR is used or not is specified in the OperatorConfigSIA.json file with flag "DOWN_SAMPLE".
# If "DOWN_SAMPLE" is true, then the DOWNSCALE_FACTOR specified in this configuration file will be used.
# If "DOWN_SAMPLE" is false, then DOWNSCALE_FACTOR of 1.0 will be used, irrespective of what it is set here.
# Note: DOWNSCALE_FACTOR should either be 1 or 0.5, with possible 0.25 if absolutely needed for performance.
#       It should never be greater than 1.
DOWNSCALE_FACTOR = 0.5

# VIS_THRESHOLD_LEVEL is a threshold used when segmenting from visual image in function
# "segment_image_set_by_vis_img()".
# This is a threshold that is applied after the background was subtracted out. When the background is effectively
# removed, only a small threshold is needed.
# We use a low threshold to make sure we identify any pixels belonging to an object.  Any artifacts that are a
# consequence of the low threshold are removed at later stage.
VIS_THRESHOLD_LEVEL = 5


# In function segment_image_set_obj_by_nir(), constant MAX_HEIGHT_TO_WIDTH_RATIO is used to filter
# out objects of extreme proportions that are likely artifacts.  MIN_OBJECT_WIDTH = 15 is used to improve
# performance so that fewer objects need to be analyzed.
MAX_HEIGHT_TO_WIDTH_RATIO = 25
MIN_OBJECT_WIDTH = 15

# Number of rows in each segmented image, if resulting segmented images are to be written out.
ROWS_PER_SEG_OUTPUT_IMG = 4000

# Define error codes (these should never be changed)
ERROR_CODE_BAD_FRAME_NUM = -200
ERROR_CODE_VIS_NIR_MISMATCH = -201
ERROR_CODE_MISSING_IMAGES = -202
ERROR_CODE_IMAGE_FOLDER_MISSING = -203
ERROR_CODE_UNABLE_TO_OPEN_IMAGE = -204
ERROR_CODE_UNABLE_TO_ALLOCATE_MEMORY_TO_IMAGE = -205
ERROR_CODE_UNABLE_TO_LOAD_IMAGE = -206
ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE = -207
ERROR_CODE_UNABLE_TO_READ_CONFIG_FILE = -208
ERROR_CODE_UNKNOWN_SEGMENT_ALGO = -209
ERROR_CODE_UNABLE_TO_CREATE_FOLDER = -210
ERROR_CODE_OUTPUT_FOLDER_INVALID = -211
ERROR_CODE_NOT_ENOUGH_SEG_IMG_ROWS = -212   # Not enough rows in segmented image to write out.

WARNING_CODE_FEWER_THAN_EXPECTED_FRAMES = -101
