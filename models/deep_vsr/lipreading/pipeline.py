# pipeline.py

import os
import shutil
from lipreading.config import args
from lipreading.data_processing.data_converter import CSVToTXTConverter
from lipreading.data_processing.label_video_splitter import LabelVideoSplitter
from lipreading.data_processing.split_files_creator import SplitFilesCreator
from lipreading.data_processing.data_preprocessor import DataPreprocessor
from lipreading.training.model_trainer import ModelTrainer

def main_pipeline():
    """
    Main function to execute the entire pipeline.
    """
    # Step 1: Convert CSV to TXT
    print("=== Step 1: Converting CSV to TXT ===")
    converter = CSVToTXTConverter()
    converter.convert_all_csv_to_txt()

    # Step 2: Split Labels and Videos
    print("\n=== Step 2: Splitting Labels and Videos ===")
    splitter = LabelVideoSplitter()
    splitter.process_all_videos_labels()

    # Step 3: Create Split Files
    print("\n=== Step 3: Creating Split Files ===")
    splitter_creator = SplitFilesCreator()
    splitter_creator.create_split_files()

    # Step 4: Move 'split_output' to 'main' and Preprocess Data
    print("\n=== Step 4: Organizing Folders and Preprocessing Data ===")
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()

    # Step 5: Train the Model with Curriculum Learning
    print("\n=== Step 5: Training the Model with Curriculum Learning ===")
    trainer = ModelTrainer()
    trainer.train_model()

    # Step 6: Cleanup Unnecessary Folders
    print("\n=== Step 6: Cleaning Up Unnecessary Folders ===")
    folders_to_delete = [
        os.path.join(args["DATA_DIRECTORY"], 'processed_csv_final'),
        os.path.join(args["DATA_DIRECTORY"], 'processed_mp4'),
    ]
    for folder in folders_to_delete:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")
            except Exception as e:
                print(f"Error deleting folder {folder}: {e}")
        else:
            print(f"Folder not found, skipping deletion: {folder}")

    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main_pipeline()
