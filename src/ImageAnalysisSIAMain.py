
#
# ImageAnalysisSIAMain.py
#
# This file contains the main function for the SIA Image Analysis software.
# This file calls methods from the ProcessImagesSIA class to load the configuration file,
# load SIA images, process the images, and save the results to a CSV files.
#
# Written by: Mark Polak
#
# Date Created: 2023-11-13
# Last Modified: 2023-11-13
#

import time
import sys
from ImageAnalysisSIA import ProcessImagesSIA


def main():

    print("Starting main function!")
    # Check if a folder path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python ImageAnalysisSIAMain.py /path/to/the/input_folder ")
        return 1

    in_folder_path = sys.argv[1]

    # Initializes an instance of the ProcessImagesSIA class with the input folder path.
    process_images_sia = ProcessImagesSIA(in_folder_path)

    # Loads configuration parameters from a JSON file using the load_json_config_file method.
    status = process_images_sia.load_json_config_file()
    if status != 0:
        return status

    # Retrieves the list of image files and verifies their existence using the get_file_list method.
    status = process_images_sia.get_file_list()
    if status != 0:
        return status

    start_time = time.time()
    # Segments the images using the segment_images method.
    status = process_images_sia.segment_images()
    if status != 0:
        return status

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nsegment_images elapsed time = ", elapsed_time)

    # If the save_segmented_images attribute is True, writes out segmented images to the output folder using the
    # write_out_segmented_images method.
    if process_images_sia.save_segmented_images:
        process_images_sia.write_out_segmented_images()

    start_time = time.time()
    # Writes CSV files containing object properties using the write_csv_files method.
    process_images_sia.write_csv_files()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nwrite_csv_file elapsed time = ", elapsed_time)

    return 0


if __name__ == "__main__":
    # Run the main function and exit with the status code which the control software can retrieve.
    sys.exit(main())

