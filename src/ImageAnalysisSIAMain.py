
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
        print("Usage: python your_program.py /path/to/your/folder")
        sys.exit(1)

    folder_path = sys.argv[1]

    print("\nFolder = ", folder_path)
    # path = r"C:\Users\markn\OneDrive\Xanantec Work\SIA\test_data\ConcatLSCAN-03208-2023-06-05-10-16-17"

    process_images_sia = ProcessImagesSIA(folder_path)

    process_images_sia.load_json_config_file()

    start_time = time.time()
    process_images_sia.get_file_list()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nget_file_list elapsed time = ", elapsed_time)
    start_time = time.time()
    process_images_sia.segment_images()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nsegment_images elapsed time = ", elapsed_time)
    start_time = time.time()
    process_images_sia.write_csv_files()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\nwrite_csv_file elapsed time = ", elapsed_time)


if __name__ == "__main__":
    main()

