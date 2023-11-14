#
# SupportUtil.py
#
# This file contains support functions used in SIA image processing software.
# Currently only write_error_to_file() is included.
#
# Written by: Mark Polak
#
# Date Created: 2023-09-21
# Last Modified: 2023-11-12
#


import datetime


def write_error_to_file(error_file, offending_file, error_code, message):
    try:
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Format the date and time as a string
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

        # Create or open the file in append mode ('a')
        with open(error_file, 'a') as file:
            # Write the message along with the formatted date and time
            file.write(f"{formatted_datetime} - '{offending_file}' - {error_code} - {message}\n")
    except Exception as e:
        print(f"An error occurred: {str(e)}  during call to write_error_to_file()")
