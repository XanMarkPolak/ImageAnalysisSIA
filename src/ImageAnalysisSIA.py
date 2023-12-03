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
import SysConfigSIA

from skimage import measure

from SupportUtil import write_error_to_file
from ImageSegmentationSIA import segment_image_set_obj_by_nir
from ImageSegmentationSIA import segment_image_set_by_vis_img
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
        self.syringe_pump_speed = 0.0
        self.segment_air_and_sand = 0.0
        self.nir_image_files = []
        self.vis_image_files = []
        self.down_sample = False
        self.down_sample_factor = SysConfigSIA.DOWNSCALE_FACTOR
        self.k1 = 2.0
        self.k2 = 4.0
        self.bad_left_edge = SysConfigSIA.BAD_EDGE_LEFT
        self.bad_right_edge = SysConfigSIA.BAD_EDGE_RIGHT
        self.min_obj_diam_um = 0.0
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
                self.segment_air_and_sand = config["SEGMENT_AIR_AND_SAND"]
                self.syringe_pump_speed = config["SYRINGE_PUMP_SPEED"]
                self.bad_left_edge = config["BAD_EDGE_LEFT"]
                self.bad_right_edge = config["BAD_EDGE_RIGHT"]
                self.down_sample = config["DOWN_SAMPLE"]
                if self.down_sample:
                    self.down_sample_factor = SysConfigSIA.DOWNSCALE_FACTOR
                else:
                    self.down_sample_factor = 1.0
                self.k1 = config["K1"]
                self.k2 = config["K2"]
                self.min_obj_diam_um = config["MIN_OBJ_DIAMETER_UM"]
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

        # Adjust result image size for Region of Interest.
        # The exclusion zones on left and right of image will be cut out.
        width_roi = width - (self.bad_left_edge + self.bad_right_edge)

        # Adjust result image sizes for possible downscaling.
        width_roi = int(width_roi * SysConfigSIA.DOWNSCALE_FACTOR)
        height = int(height * SysConfigSIA.DOWNSCALE_FACTOR)

        try:
            # Attempt to create large 8-bit images for segmented object for entire experiment.  One large
            # image for bitumen objects and the other for non-bitumen objects.
            self.experiment_bitumen_binary_img = np.zeros((height * num_of_image_sets, width_roi), dtype=np.uint8)
            self.experiment_other_binary_img = np.zeros((height * num_of_image_sets, width_roi), dtype=np.uint8)
        except MemoryError as err:
            write_error_to_file(SysConfigSIA.ERROR_LOG_FILE, __file__,
                                SysConfigSIA.ERROR_CODE_UNABLE_TO_ALLOCATE_MEMORY_TO_IMAGE,
                                f"Memory allocation for the large image failed. Not enough memory available. "
                                f"Reason: {err}")
            return SysConfigSIA.ERROR_CODE_UNABLE_TO_ALLOCATE_MEMORY_TO_IMAGE

        # Iterate through all the loaded images
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

            # If "down_sample" option was chosen, the images will be resized to the given downscale factor.
            # For example, a downscale factor of 0.5 would reduce the width and height by 50%.
            if self.down_sample:
                vis_height, vis_width, _ = image_vis.shape
                nir_height, nir_width = image_nir.shape

                downscale_factor = self.down_sample_factor  # downscale_factor is used to resize image to a smaller one.
                # Resize the input images according to DOWNSCALE_FACTOR
                image_vis = cv2.resize(image_vis, (int(vis_width * downscale_factor),
                                                   int(vis_height * downscale_factor)))
                image_nir = cv2.resize(image_nir, (int(nir_width * downscale_factor),
                                                   int(nir_height * downscale_factor)))

            target_row = height * img_num  # height has already been adjusted by DOWNSCALE_FACTOR

            # Segment the image set and identify objects of interest in each image.
            # A binary image for bitumen (dark) objects and a binary image for air/sand (light) objects are
            # returned.  The returned images have values of 255 for object pixels and 0 for background pixels.
            self.experiment_bitumen_binary_img[target_row:target_row + height, :], \
                self.experiment_other_binary_img[target_row:target_row + height, :] = \
                segment_image_set_obj_by_nir(image_vis, image_nir, self.k1, self.k2)
            #                    segment_image_set_by_vis_img(image_vis, image_nir, self.k1, self.k2)

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
                                     csv_file, self.image_scale, self.line_scan_rate, self.syringe_pump_speed,
                                     self.calc_summary_stats, summary_csv_file)


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


def get_object_properties_from_binary_image(binary_image, obj_type, image_scale, line_scan_rate, downscaled_factor,
                                            pump_speed):
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

    ski_props_list = []

    for obj_props in props:
        major = obj_props.major_axis_length
        minor = obj_props.minor_axis_length

        # Equations were originally written with orientation 0 being from horizontal axis.  In regionprops(),
        # orientation of 0 is vertical.  Easiest fix was to add PI/2 to the orientation.
        fixed_orient = obj_props.orientation + 1.570796327

        min_row, min_col, max_row, max_col = obj_props.bbox

        # Calculate height
        height = (max_row - min_row)

        diameter = image_scale * major * minor / np.sqrt(
            (minor * np.cos(fixed_orient)) ** 2 + (major * np.sin(fixed_orient)) ** 2)

        # Calculate the speed of the object in mm/s, and subtract out speed of the fluid controlled by the syringe pump.
        speed = (line_scan_rate * diameter / (height * 1000)) - pump_speed

        # If object type is passed in as unknown (UNK), then figure out if it is air bubble or sand, based on
        # speed of the object.
        if obj_type == "BIT":
            obj_type_this_obj = "BIT"
        else:
            if speed > 0.0:
                obj_type_this_obj = "AIR"  # If non-bitumen object is rising in fluid, then it must be air.
            else:
                obj_type_this_obj = "SND"  # If non-bitumen object is falling in fluid, then it must be sand.

        ski_props_list.append([
            obj_props.centroid[1] / downscaled_factor,
            obj_props.centroid[0] / downscaled_factor,
            obj_props.area / downscaled_factor,
            speed,
            obj_type_this_obj,
            major / downscaled_factor,
            minor / downscaled_factor,
            obj_props.orientation,
            obj_props.solidity,
            obj_props.eccentricity,
            diameter / downscaled_factor
        ])
    return ski_props_list


def object_properties_to_csv(binary_bitumen_image, binary_non_bitumen_image, csv_file, image_scale, line_scan_rate,
                             pump_speed, create_summary_stats, summary_csv_file=None):
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

    downscale_factor = SysConfigSIA.DOWNSCALE_FACTOR

    # Get the object properties for the binary image of bitumen objects and light colored objects passed in.
    bitumen_props_list = get_object_properties_from_binary_image(binary_bitumen_image, "BIT", image_scale,
                                                                 line_scan_rate, downscale_factor, pump_speed)
    other_obj_props_list = get_object_properties_from_binary_image(binary_non_bitumen_image, "UNK", image_scale,
                                                                   line_scan_rate, downscale_factor, pump_speed)

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
            writer_obj.writerows(other_obj_props_list)
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
    print("create_summary_stats= ", create_summary_stats, " summary_csv_file=",summary_csv_file)
    if create_summary_stats and summary_csv_file is not None:
        num_of_bit_obj = len(bitumen_props_list)

        # Assuming bitumen_props_list has diameters as the last element, extract all the diameter measurement for
        # bitumen (bit) objects.
        diameters_bit = [obj[-1] for obj in bitumen_props_list]
        avg_diameter_bit = np.mean(diameters_bit)

        # For ObjClass "AIR", which other_obj_props_list has as the index=4, extract all the diameter measurements
        # where the diameters are the last element.
        diameters_air = [obj[-1] for obj in other_obj_props_list if obj[4] == "AIR"]
        avg_diameter_air = np.mean(diameters_air)

        # For ObjClass "SND", which other_obj_props_list has as the index=4, extract all the diameter measurements
        # where the diameters are the last element.
        diameters_snd = [obj[-1] for obj in other_obj_props_list if obj[4] == "SND"]
        avg_diameter_snd = np.mean(diameters_snd)

        # Assuming other_obj_props_list has speed as the index=3 element, extract all the speed measurements.
        speeds_bit = [obj[3] for obj in bitumen_props_list]
        avg_speed_bit = np.mean(speeds_bit)

        # For ObjClass "AIR", which other_obj_props_list has as the index=4, extract all the speed measurements
        # which are at index 3.
        speed_air = [obj[3] for obj in other_obj_props_list if obj[4] == "AIR"]
        avg_speed_air = np.mean(speed_air)

        # For ObjClass "SND", which other_obj_props_list has as the index=4, extract all the speed measurements
        # which are at index 3.
        speed_snd = [obj[3] for obj in other_obj_props_list if obj[4] == "SND"]
        avg_speed_snd = np.mean(speed_snd)

        # Convert diameters to a sorted NumPy array
        sorted_diameters = np.sort(diameters_bit)

        # Calculate cumulative sum
        cpp_sum_num = np.cumsum(np.ones(num_of_bit_obj))
        cpp_sum_diam = np.cumsum(sorted_diameters)
        cpp_sum_vol = np.cumsum(sorted_diameters**3)

        # Calculate cumulative percent passing
        cpp_number = 100 * cpp_sum_num / num_of_bit_obj
        cpp_diameter = 100 * cpp_sum_diam / np.sum(sorted_diameters)
        cpp_volume = 100 * cpp_sum_vol / np.sum(sorted_diameters**3)

        try:
            with open("tmp_sum.txt", 'w', newline='') as file_sum:
                writer_sum = csv.writer(file_sum)

                writer_sum.writerow(["NumOfBitObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([num_of_bit_obj, avg_diameter_bit, avg_speed_bit])
                writer_sum.writerow(["NumOfSandObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([len(diameters_snd), avg_diameter_snd, avg_speed_snd])
                writer_sum.writerow(["NumOfAirObjects", "AvgDiameter", "AvgSpeed"])
                writer_sum.writerow([len(diameters_air), avg_diameter_air, avg_speed_air])

                writer_sum.writerow(["Diameter(um)", "CPP_by_Number(%)", "CPP_by_Diameter(%)", "CPP_by_Volume(%)"])

                # Write the data for Cumulative Percent Passing by Diameter
                for row in range(num_of_bit_obj):
                    writer_sum.writerow([sorted_diameters[row], cpp_number[row],  cpp_diameter[row], cpp_volume[row]])
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
