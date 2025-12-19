# split_files_creator.py

import os
import random
from lipreading.config import args

class SplitFilesCreator:
    def __init__(self):
        self.folder_path = args["DATA_DIRECTORY"]
        self.split_output_folder = os.path.join(self.folder_path, 'split_output')
        self.split_proportions = (0.7, 0.15, 0.15)  # (train, val, test)
    
    def create_split_files(self):
        """
        Creates split files (train.txt, val.txt, test.txt, pretrain.txt) for training, validation, and testing.
        """
        if not os.path.exists(self.split_output_folder):
            print(f"Folder {self.split_output_folder} does not exist.")
            return
        
        mp4_files = [f for f in os.listdir(self.split_output_folder) if f.endswith('.mp4')]
        
        if not mp4_files:
            print("No .mp4 files found in the split_output folder.")
            return

        # Remove the ".mp4" extension and create a list of relative paths
        file_list = [os.path.splitext(f)[0] for f in mp4_files]

        # Shuffle the list to ensure randomness
        random.shuffle(file_list)

        # Split the list according to the specified proportions
        total_files = len(file_list)
        train_split = int(self.split_proportions[0] * total_files)
        val_split = int(self.split_proportions[1] * total_files)
        test_split = total_files - train_split - val_split

        train_files = file_list[:train_split]
        val_files = file_list[train_split:train_split + val_split]
        test_files = file_list[train_split + val_split:]

        # Function to write a list of files to a text file
        def write_to_file(filename, data):
            with open(os.path.join(self.folder_path, filename), 'w', encoding='utf-8') as file:
                for item in data:
                    file.write(f"split_output\\{item}\n")
            print(f"File {filename} has been created successfully.")

        # Write to each of the required files
        write_to_file('pretrain.txt', file_list)
        write_to_file('train.txt', train_files)
        write_to_file('val.txt', val_files)
        write_to_file('test.txt', test_files)
