# data_converter.py

import os
import pandas as pd
from lipreading.config import args

class CSVToTXTConverter:
    def __init__(self):
        self.csv_folder = os.path.join(args["DATA_DIRECTORY"], 'processed_csv_final')
        self.output_folder = os.path.join(args["DATA_DIRECTORY"], 'processed_mp4')
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
    def convert_csv_to_txt(self, csv_file, txt_file):
        """
        Converts a single CSV file to TXT format.
        """
        df = pd.read_csv(csv_file)
        
        # Create the text content
        text_content = "Text: " + " ".join(df['Word'].tolist()).upper() + "\n"
        text_content += "Conf:  1\n\n"
        text_content += "WORD START END\n"

        for index, row in df.iterrows():
            text_content += f"{row['Word'].upper()} {row['Start']} {row['End']}\n"

        # Write to TXT file with UTF-8 encoding
        with open(txt_file, 'w', encoding='utf-8') as file:
            file.write(text_content)
        
    def convert_all_csv_to_txt(self):
        """
        Converts all CSV files in the csv_folder to TXT format.
        """
        for csv_file in os.listdir(self.csv_folder):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(self.csv_folder, csv_file)
                txt_file_name = os.path.splitext(csv_file)[0] + '.txt'
                txt_path = os.path.join(self.output_folder, txt_file_name)
                self.convert_csv_to_txt(csv_path, txt_path)
                print(f"Converted {csv_file} to {txt_file_name}")
