import os
import sys


# function to extract file using tar command
def extract_data(file_path, extract_path):
    os.system('tar -xvf ' + file_path + ' -C ' + extract_path)


if __name__ == '__main__':
    data_root = r"F:\Code\datasets\train_check"
    # loop through all the files in the data_root directory
    for file in os.listdir(data_root):
        # check if the file is a tar file
        if file.endswith('.tar.gz'):
            # extract the file
            print(f"Extracting {file}")
            extract_data(os.path.join(data_root, file), data_root)
            print(f"Extracted {file}")
            # Remove the tar file
            os.remove(os.path.join(data_root, file))
        else:
            print(f"File {file} is not a tar file")