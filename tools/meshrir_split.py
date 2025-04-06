import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', default=None, required=True,
                    help='/path/to/parent/folder')      # option that takes a value
args = parser.parse_args()

# Define the path to the dataset and the target training/validation directories
base_folder = args.base_folder
dataset_path = os.path.join(base_folder, 'S1-M3969_npy')
train_path = os.path.join(base_folder, 'train')
test_path = os.path.join(base_folder, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# List all .npy files in the dataset path
files = [d for d in os.listdir(dataset_path) if 'ir' in d]
# pos_mic.npy and pos_src.npy files
pos_files = [d for d in os.listdir(dataset_path) if 'pos' in d]

random.shuffle(files)

# Define the split ratio (e.g., 80% training, 20% validation)
split_ratio = 0.9
split_index = int(len(files) * split_ratio)

# Split directories into training and validation sets
train_data = files[:split_index]
test_data = files[split_index:]

# Function to copy directories to a new location


def copy_directories(data, source_path, dest_path):
    for d in data:
        source_file = os.path.join(source_path, d)
        destination_file = os.path.join(dest_path, d)
        if not os.path.exists(destination_file):
            # Use copy2 to preserve metadata
            shutil.copy2(source_file, destination_file)
        else:
            print(f"File {destination_file} already exists. Skipping copy.")


# Copy data to the corresponding paths
copy_directories(train_data, dataset_path, train_path)
copy_directories(test_data, dataset_path, test_path)
copy_directories(pos_files, dataset_path, base_folder)

print("Dataset splitting complete.")
