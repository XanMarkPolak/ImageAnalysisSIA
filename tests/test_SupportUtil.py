
import os
import pytest
import datetime
from SupportUtil import write_error_to_file  # Import the function to be tested


# Define a temporary test file path for testing
TEST_FILE_PATH = r".\test_errors.log"

# Fixture to set up and tear down the test environment
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Remove the test file if it exists before each test
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)


# Test cases using pytest
def test_write_error_to_file():
    # Test data
    error_code = "001"
    message = "Test error message"

    # Call the function to write an error message
    write_error_to_file(TEST_FILE_PATH, __file__, error_code, message)

    # Verify that the file has been created
    assert os.path.exists(TEST_FILE_PATH)

    # Read the content of the file
    with open(TEST_FILE_PATH, 'r') as file:
        lines = file.readlines()

    # Verify that the file contains the expected log entry
    expected_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    expected_entry = f"{expected_datetime} - '{__file__}' - {error_code} - {message}\n"
    assert expected_entry in lines


