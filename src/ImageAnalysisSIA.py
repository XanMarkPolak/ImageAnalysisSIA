
import os
import numpy as np
import cv2

# Defined constants.
FIRST_FRAME_CHARACTER = 30  # first character in image filename that specifies frame number.
LAST_FRAME_CHARACTER = 34   # last character in image filename that specifies frame number.


class ProcessImagesSIA:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.nir_image_files = []
        self.vis_image_files = []

    def get_file_list_and_verify_correct_files_exist(self):
        nir_image_files = [f for f in os.listdir(self.image_folder) if f.startswith('LS-NIR') and f.endswith('.tif')]
        vis_image_files = [f for f in os.listdir(self.image_folder) if f.startswith('LS-VIS') and f.endswith('.tif')]

        # Define a sorting function based on the frame number
        def custom_sort(file_name):
            try:
                frame_num = int(file_name[FIRST_FRAME_CHARACTER:LAST_FRAME_CHARACTER])
                return frame_num
            except ValueError as e:
                # Handle the error here (e.g., print an error message, log it, or raise a custom exception)
                print(f"Error: Unable to convert frame number '{file_name[FIRST_FRAME_CHARACTER:LAST_FRAME_CHARACTER]}' in file '{file_name}' to an integer. Reason: {e}")
                # TODO: Wrtite error to file.

            return frame_num

        # Sort the list using the  sorting function
        nir_image_files.sort(key=custom_sort)


        print("\n")
        for f in nir_image_files:
            print(f)
        for f in vis_image_files:
            print(f)




# Create a memory-mapped array for the image
#image = np.memmap('large_image.dat', dtype='uint8', mode='r+', shape=(10000000, 4096))
