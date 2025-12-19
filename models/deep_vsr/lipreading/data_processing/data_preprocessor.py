# data_preprocessor.py

import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
from lipreading.config import args
from lipreading.models.visual_frontend import VisualFrontend
from lipreading.utils.preprocessing import preprocess_sample

class DataPreprocessor:
    def __init__(self):
        self.data_directory = args["DATA_DIRECTORY"]
        self.main_folder = os.path.join(self.data_directory, 'main')
        self.split_output_folder = os.path.join(self.data_directory, 'split_output')
        self.processed_mp4_folder = os.path.join(self.main_folder, 'split_output')  # After moving
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
    
    def move_split_output_to_main(self):
        """
        Moves the 'split_output' folder into the 'main' directory.
        """
        if os.path.exists(self.processed_mp4_folder):
            print(f"'split_output' is already inside '{self.main_folder}'. Skipping move.")
            return
        if not os.path.exists(self.split_output_folder):
            print(f"'split_output' folder does not exist at {self.split_output_folder}.")
            return
        shutil.move(self.split_output_folder, self.main_folder)
        print(f"Moved 'split_output' to '{self.main_folder}'.")
    
    def preprocess_data(self):
        """
        Preprocesses the data by extracting and normalizing visual features.
        """
        # Move 'split_output' into 'main' before preprocessing
        self.move_split_output_to_main()
        
        np.random.seed(args["SEED"])
        torch.manual_seed(args["SEED"])
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda" if gpu_available else "cpu")

        # Declaring the visual frontend module
        vf = VisualFrontend()
        vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
        vf.to(device)

        # Walking through the data directory and obtaining a list of all files in the dataset
        files_list = []
        for root, dirs, files in os.walk(self.processed_mp4_folder):
            for file in files:
                if file.endswith(".mp4"):
                    files_list.append(os.path.join(root, file[:-4]))

        # Preprocessing each sample
        print(f"\nNumber of data samples to be processed = {len(files_list)}")
        print("\n\nStarting preprocessing ....\n")

        params = {
            "roiSize": args["ROI_SIZE"],
            "normMean": args["NORMALIZATION_MEAN"],
            "normStd": args["NORMALIZATION_STD"],
            "vf": vf
        }

        for file in tqdm(files_list, leave=True, desc="Preprocess", ncols=75):
            preprocess_sample(file, params)

        print("\nPreprocessing Done.")

        # Generating preval.txt for splitting the pretrain set into train and validation sets
        print("\n\nGenerating the preval.txt file ....")

        pretrain_txt_path = os.path.join(self.data_directory, "pretrain.txt")
        preval_txt_path = os.path.join(self.data_directory, "preval.txt")

        with open(pretrain_txt_path, "r") as f:
            lines = f.readlines()

        if os.path.exists(preval_txt_path):
            with open(preval_txt_path, "r") as f:
                lines.extend(f.readlines())

        indices = np.arange(len(lines))
        np.random.shuffle(indices)
        val_indices = np.sort(indices[:int(np.ceil(args["PRETRAIN_VAL_SPLIT"] * len(indices)))])
        train_indices = np.sort(indices[int(np.ceil(args["PRETRAIN_VAL_SPLIT"] * len(indices))):])

        lines = np.sort(np.array(lines))
        with open(pretrain_txt_path, "w") as f:
            f.writelines(list(lines[train_indices]))
        with open(preval_txt_path, "w") as f:
            f.writelines(list(lines[val_indices]))

        print("\npreval.txt file generated.\n")
