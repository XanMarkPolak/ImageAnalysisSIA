#
# ImageAnalysisSIA.py
#
# This is the main file that defines the class ProcessImagesSIA and its methods for loading configuration file,
# loading SIA images, processing the images, and saving the results in CSV files.
#
# Written by: Mark Polak
#
# Date Created: 2023-09-18
# Last Modified: 2023-11-12
#


import os
import csv
import numpy as np
import cv2
import json
import time
from skimage import measure

from SupportUtil import write_error_to_file
from ImageSegmentationSIA import segment_image_set_obj_by_nir
import SysConfigSIA


class ProcessImagesSIA:
    experiment_bitumen_binary_img = None
    experiment_other_binary_img = None  # Binary image for non-bitumen objects (air, sand, unknown)

    # Config Loaded from JSON file
    image_scale = 0
    line_scan_rate = 0
    save_segmented_images = False
    calc_summary_stats = False

    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.start_frame_character = SysConfigSIA.FIRST_IMAGE_NAME_FRAME_CHARACTER
        self.end_frame_character = SysConfigSIA.LAST_IMAGE_NAME_FRAME_CHARACTER
        self.first_valid_frame = SysConfigSIA.FIRST_FRAME_NUM
        self.last_valid_frame = SysConfigSIA.LAST_FRAME_NUM
        self.bad_left_edge = SysConfigSIA.BAD_EDGE_LEFT
        self.bad_right_edge = SysConfigSIA.BAD_EDGE_RIGHT
        self.nir_image_files = []
        self.vis_image_files = []
        self.config_json_loaded = False

    # Define methods
    def load_json_config_file(self):
        file_path = SysConfigSIA.OPERATOR_CONFIG_FILE_JSON
        try:
            with open(file_path, "r") as json_file:
                config = json.load(json_file)
                self.image_scale = config["IMAGE_SCALE"]
                self.line_scan_rate = config["LINE_SCAN_RATE"]
                self.save_segmented_images = config["SAVE_SEGMENTED_IMAGES"]
                self.calc_summary_stats = config["CALC_SUMMARY_STATS"]
                self.config_json_loaded = True
        except FileNotFoundError:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__,
                                SysConfigSIA.ERROR_CODE_UNABLE_TO_READ_CONFIG_FILE,
                                f"Error reading configuration file. File {file_path} not found.")
        except json.JSONDecodeError as err:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__,
                                SysConfigSIA.ERROR_CODE_UNABLE_TO_READ_CONFIG_FILE,
                                f"Error reading configuration file. Error decoding JSON."
                                f"Reason: {err}")

    def get_file_list(self):
        self.vis_image_files, self.nir_image_files = get_file_list_and_verify_correct_files_exist(
            self.image_folder, self.start_frame_character, self.end_frame_character, self.first_valid_frame,
            self.last_valid_frame)

    def segment_images(self):
        num_of_image_sets = len(self.vis_image_files)

        # Use the cv2.IMREAD_UNCHANGED flag to load the image without decoding it
        img_info = cv2.imread(self.image_folder + "\\" + self.vis_image_files[0], cv2.IMREAD_UNCHANGED)
        if img_info is not None:
            # Get the width and height of the image
            width, height = img_info.shape[1], img_info.shape[0]
        else:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_OPEN_IMAGE,
                                f"Unable open image '{self.vis_image_files[0]}' to read dimensions. "
                                f"Unknown Reason")
            return SysConfigSIA.ERROR_CODE_UNABLE_TO_OPEN_IMAGE

        width_roi = width - (self.bad_left_edge + self.bad_right_edge)
        try:
            # Attempt to create large 8-bit images for segmented object for entire experiment.  One large
            # image for bitumen objects and the other for non-bitumen objects.
            self.experiment_bitumen_binary_img = np.zeros((height * num_of_image_sets, width_roi), dtype=np.uint8)
            self.experiment_other_binary_img = np.zeros((height * num_of_image_sets, width_roi), dtype=np.uint8)
            print("Memory allocation for the large image was successful.")
        except MemoryError as err:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__,
                                SysConfigSIA.ERROR_CODE_UNABLE_TO_ALLOCATE_MEMORY_TO_IMAGE,
                                f"Memory allocation for the large image failed. Not enough memory available. "
                                f"Reason: {err}")
            return SysConfigSIA.ERROR_CODE_UNABLE_TO_ALLOCATE_MEMORY_TO_IMAGE

        for img_num in range(num_of_image_sets):
            print("Processing Img", img_num)
            start_time = time.time()

            vis_file_path = self.image_folder + "\\" + self.vis_image_files[img_num]
            nir_file_path = self.image_folder + "\\" + self.nir_image_files[img_num]
            img_vis_full = cv2.imread(vis_file_path)
            img_nir_full = cv2.imread(nir_file_path, cv2.IMREAD_GRAYSCALE)
            if img_vis_full is None or img_nir_full is None:
                write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_LOAD_IMAGE,
                                    f"Failed to load image set: "
                                    f"{vis_file_path} and {nir_file_path}")
                return SysConfigSIA.ERROR_CODE_UNABLE_TO_LOAD_IMAGE

            # Cut out the region of interest (ROI)
            image_vis = img_vis_full[:, SysConfigSIA.BAD_EDGE_LEFT:(width - SysConfigSIA.BAD_EDGE_RIGHT), :]
            image_nir = img_nir_full[:, SysConfigSIA.BAD_EDGE_LEFT:(width - SysConfigSIA.BAD_EDGE_RIGHT)]

            target_row = height * img_num
            self.experiment_bitumen_binary_img[target_row:target_row + height, :], \
                self.experiment_other_binary_img[target_row:target_row + height, :] = \
                segment_image_set_obj_by_nir(image_vis, image_nir)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("        Image", img_num, "elapsed time = ", elapsed_time)
        cv2.imwrite("out_dark.png", self.experiment_bitumen_binary_img)
        cv2.imwrite("out_light.png", self.experiment_other_binary_img)

    def write_csv_files(self):
        # Configuration file must be loaded in order to calculate object properties in real units that are written
        # to the CSV file.   If config file was not loaded, then exit without creating a CSV file.
        if self.config_json_loaded:
            # save results to csv file
            fn_str = self.nir_image_files[0][
                     SysConfigSIA.FIRST_IMAGE_NAME_SAMPLE_CHARACTER:SysConfigSIA.LAST_IMAGE_NAME_SAMPLE_CHARACTER + 1]
            csv_file = "LSCAN-Res-" + fn_str + "-Objects.csv"
            summary_csv_file = "LSCAN-Res-" + fn_str + "-Summary.csv"
            object_properties_to_csv(self.experiment_bitumen_binary_img, self.experiment_other_binary_img,
                                     csv_file, self.image_scale, self.line_scan_rate, self.calc_summary_stats,
                                     summary_csv_file)


def get_file_list_and_verify_correct_files_exist(image_folder, start_frame_character, end_frame_character,
                                                 first_valid_frame, last_valid_frame):
    """
    Retrieve and verify lists of VIS and NIR image files in a given folder, and perform various checks.

    Parameters:
    image_folder (str): Path to the folder containing image files.
    start_frame_character (int): The index of the first character of the frame number in the image file names.
    end_frame_character (int): The index of the last character of the frame number in the image file names.
    first_valid_frame (int): The first valid frame number, or -1 for no restriction.
    last_valid_frame (int): The last valid frame number, or -1 for no restriction.

    Returns:
    tuple: A tuple containing two lists - the first list is VIS image files, and the second list is NIR image files.
    If any errors occur during the process, None is returned for both lists.
    """

    # Define a custom sorting function for image files based on frame numbers.
    # Parameters:
    #   file_name (str): The name of the image file.
    # Returns:
    #   int: The frame number extracted from the file name.
    def custom_sort(file_name):
        try:
            frame_num = int(file_name[start_frame_character:end_frame_character])
            return frame_num
        except ValueError as err:
            # Handle the error
            frm_str = file_name[start_frame_character:end_frame_character]
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_BAD_FRAME_NUM,
                                f"Unable to convert frame number '{frm_str}' in file '{file_name}' to an integer. "
                                f"Reason: {err}")
            return None

    # Define a support function to check if a frame number extracted from the image file name is within the valid range.
    # Parameters:
    #   file_name (str): The name of the image file.
    # Returns:
    #   bool: True if the frame number is within the valid range, otherwise False.
    def frame_in_range(file_name):
        try:
            frame_num = int(file_name[start_frame_character:end_frame_character + 1])
        except ValueError as err:
            # Handle the error
            frm_str = file_name[start_frame_character:end_frame_character]
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_BAD_FRAME_NUM,
                                f"Unable to convert frame number '{frm_str}' in file '{file_name}' to an integer. "
                                f"Reason: {err}")
            return False

        # Check if the frame number is within the valid range or if there are no restrictions (-1)
        if (frame_num >= first_valid_frame or first_valid_frame == -1) and \
                (frame_num <= last_valid_frame or last_valid_frame == -1):
            return True
        else:
            return False

    # Check if the folder exists
    if os.path.exists(image_folder):
        nir_image_files = [f for f in os.listdir(image_folder) if
                           f.startswith('LS-NIR-') and frame_in_range(f)]
        vis_image_files = [f for f in os.listdir(image_folder) if
                           f.startswith('LS-VIS-') and frame_in_range(f)]
    else:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_IMAGE_FOLDER_MISSING,
                            f"Image folder '{image_folder}' not found.")
        return None, None

    if len(nir_image_files) < 1 or len(vis_image_files) < 1:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_MISSING_IMAGES,
                            f"VIS or NIR images are missing.")
        return None, None

    # Sort the list using the sorting function
    try:
        nir_image_files.sort(key=custom_sort)
    except Exception as e:
        # Handle the error raised by the sort operation
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_BAD_FRAME_NUM,
                            f"Unable to sort NIR image file names. Exception: {e}")
        return None, None

    try:
        vis_image_files.sort(key=custom_sort)
    except Exception as e:
        # Handle the error raised by the sort operation
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_BAD_FRAME_NUM,
                            f"Unable to sort VIS image file names. Exception: {e}")
        return None, None

    # Make sure same number of NIR and VIS files
    if len(vis_image_files) != len(nir_image_files):
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_VIS_NIR_MISMATCH,
                            f"Different number of NIR and VIS images.")
        return None, None

    # Make sure frame numbers match
    match = True
    for i in range(len(vis_image_files)):
        vis_frm_num = vis_image_files[i][start_frame_character:end_frame_character + 1]
        nir_frm_num = nir_image_files[i][start_frame_character:end_frame_character + 1]
        if vis_frm_num != nir_frm_num:
            match = False
    if not match:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_VIS_NIR_MISMATCH,
                            f"Frame numbers of VIS and NIR images do not match.")
        return None, None

    # If the last frame number in the read files is less than the last frame in the valid frame number specified
    #  in configuration, than write a warning, but continue processing.
    frm = vis_image_files[-1][start_frame_character:end_frame_character + 1]
    vis_frm_num = int(frm)  # last frame number
    if vis_frm_num < last_valid_frame:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.WARNING_CODE_FEWER_THAN_EXPECTED_FRAMES,
                            f"Warning: The last frame read is lower than the last frame specified.  "
                            f"Continuing process.")

    return vis_image_files, nir_image_files


def get_object_properties_from_binary_image(binary_image, obj_type, image_scale, line_scan_rate):
    """
    Calculate properties of objects in a binary image.
    This is a support function that is called from object_properties_to_csv().

    Parameters:
    - binary_image (ndarray): Binary image containing segmented objects.
    - obj_type (str): Type of the object (e.g., 'BIT' (bitumen), 'SND' (sand), etc.).
    - image_scale (float): Scaling factor to convert pixel dimensions to physical dimensions (um).
    - line_scan_rate (float): Line scan rate in pixels per second.

    Returns:
    list: A list containing properties of each object in the following format:
    [
        [centroid_x, centroid_y, area, speed, obj_type, major_axis, minor_axis, orientation,
         solidity, eccentricity, diameter]
    ]
    """

    # Label the objects in the binary image
    labeled_image = measure.label(binary_image, connectivity=2)

    # Calculate region properties using scikit-image's regionprops
    props = measure.regionprops(labeled_image)

    speed_constant = line_scan_rate / 1000

    ski_props_list = []

    for obj_props in props:
        major = obj_props.major_axis_length
        minor = obj_props.minor_axis_length

        # Equations were originally written with orientation 0 being from horizontal axis.  In regionprops(),
        # orientation of 0 is vertical.  Easiest fix was to add PI/2 to the orientation.
        fixed_orient = obj_props.orientation + 1.570796327

        min_row, min_col, max_row, max_col = obj_props.bbox

        # Calculate height
        height = max_row - min_row

        diameter = image_scale * major * minor / np.sqrt(
            (minor * np.cos(fixed_orient)) ** 2 + (major * np.sin(fixed_orient)) ** 2)

        speed = speed_constant * diameter / height

        ski_props_list.append([
            obj_props.centroid[1],
            obj_props.centroid[0],
            obj_props.area,
            speed,
            obj_type,
            major,
            minor,
            obj_props.orientation,
            obj_props.solidity,
            obj_props.eccentricity,
            diameter
        ])
    return ski_props_list


def object_properties_to_csv(binary_bitumen_image, binary_non_bitumen_image, csv_file, image_scale, line_scan_rate,
                             create_summary_stats, summary_csv_file=None):
    """
     Extract object properties from binary images and write the results to CSV files.

     Parameters:
     - binary_bitumen_image (ndarray): Binary image containing segmented bitumen objects.
     - binary_non_bitumen_image (ndarray): Binary image containing segmented non-bitumen (sand, air, unknown) objects.
     - csv_file (str): Path to the target CSV file to store the object properties.
     - image_scale (float): Scaling factor to convert pixel dimensions to physical dimensions (um/pix).
     - line_scan_rate (float): Line scan rate in pixels per second.
     - create_summary_stats (bool): Flag indicating whether to calculate and write summary statistics.
     - summary_csv_file (str, optional): Path to the target CSV file for summary statistics.

     Returns:
     None

     Notes:
     - The function creates a temporary file ('tmp.txt') initially, to avoid the SIA control system reading a partially
        written CSV file in case of a program crash, and then renames the 'tmp.txt' file to the specified CSV file
        name.
     - Object properties are calculated by calling the 'get_object_properties_from_binary_image()' function.
     - Summary statistics include the number of objects, average diameter, and average speed for each object type.
     - Summary statistics include cumulative percent passing by diameter for bitumen objects.
     - Summary statistics are written to a separate temporary file ('tmp_sum.txt') and then renamed to the
        specified CSV file name.

     """

    if binary_bitumen_image is None or binary_non_bitumen_image is None:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  Unable to write results to CSV file.  "
                            f"Binary image set not passed to object_properties_to_csv() .")
        return
    if csv_file == "":
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  Unable to write results to CSV file.  "
                            f"CSV file name not passed to object_properties_to_csv() .")
        return

    # Get the object properties for the binary image of bitumen objects and light colored objects passed in.
    bitumen_props_list = get_object_properties_from_binary_image(binary_bitumen_image, "BIT", image_scale,
                                                                 line_scan_rate)
    other__obj_props_list = get_object_properties_from_binary_image(binary_non_bitumen_image, "UNK", image_scale,
                                                                    line_scan_rate)

    # Initially write the results to temporary file "tmp.txt".  This temporary file is created instead of the
    # properly named target CSV file so that in case the program crashes or is killed, the control software
    # does not try to read a partially written CVS file thinking the image analysis was completed successfully.
    try:
        with open("tmp.txt", 'w', newline='') as file_obj:
            writer_obj = csv.writer(file_obj)

            # Write the header row
            writer_obj.writerow(
                ["CentroidX", "CentroidY", "Area(pix)", "Speed(mm/s)", "ObjClass", "MajorAxisLength(pix)",
                 "MinorAxisLength(pix)", "Orientation(rad)", "Solidity", "Eccentricity", "Diameter(um)"])

            # Write the data
            writer_obj.writerows(bitumen_props_list)
            writer_obj.writerows(other__obj_props_list)
    except Exception as e:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  Unable to write results to CSV file.  "
                            f"Exception {e} occurred trying to write results to tmp.txt")

    # Rename 'tmp.txt' to the correct CSV file name.
    try:
        os.rename("tmp.txt", csv_file)
    except FileNotFoundError:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  Unable to find file tmp.txt")
    except FileExistsError:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  File '{csv_file}' already exists while trying to rename tmp.txt to {csv_file}.")
    except Exception as e:
        write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                            f"Error.  Exception {e} occurred while trying to move tmp.txt to {csv_file}")

    #
    # Check if summary statistics are to be calculated and written to a CSV file.
    #
    if create_summary_stats and summary_csv_file is not None:
        num_of_bit_obj = len(bitumen_props_list)

        # Assuming ski_props_list has diameters as the last element, extract all the diameter measurement.
        diameters = [obj[-1] for obj in bitumen_props_list]
        avg_diameter = np.mean(diameters)

        # Assuming ski_props_list has speed as the index=3 element, extract all the diameter measurement.
        speeds = [obj[3] for obj in bitumen_props_list]
        avg_speed = np.mean(speeds)

        # Convert diameters to a sorted NumPy array
        sorted_diameters = np.sort(diameters)

        # Calculate cumulative sum
        cpp_sum = np.cumsum(sorted_diameters)

        # Calculate cumulative percentage
        cpp_diameter = 100 * cpp_sum / np.sum(sorted_diameters)

        try:
            with open("tmp_sum.txt", 'w', newline='') as file_sum:
                writer_sum = csv.writer(file_sum)

                writer_sum.writerow(["NumOfBitObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([num_of_bit_obj, avg_diameter, avg_speed])
                writer_sum.writerow(["NumOfSandObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([0, 0.0, 0.0])
                writer_sum.writerow(["NumOfAirObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([0, 0.0, 0.0])
                writer_sum.writerow(["NumOfUnKnObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([0, 0.0, 0.0])

                writer_sum.writerow(["Diameter(um)", "CPP_by_Diameter)"])

                # Write the data for Cumulative Percent Passing by Diameter
                for row in range(num_of_bit_obj):
                    writer_sum.writerow([sorted_diameters[row], cpp_diameter[row]])

        except Exception as e:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                                f"Error.  Unable to write summary results to CSV file.  "
                                f"Exception {e} occurred trying to write results to tmp.txt")

        # Rename 'tmp_sum.txt' to the correct CSV file name.
        try:
            os.rename("tmp_sum.txt", summary_csv_file)
        except FileNotFoundError:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                                f"Error.  Unable to find file tmp_sum.txt")
        except FileExistsError:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                                f"Error.  File '{summary_csv_file}' already exists while trying to rename tmp_sum.txt "
                                f"to {summary_csv_file}.")
        except Exception as e:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__, SysConfigSIA.ERROR_CODE_UNABLE_TO_WRITE_CSV_FILE,
                                f"Error.Exception {e} occurred while trying to move tmp_sum.txt to {summary_csv_file}")
