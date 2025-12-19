# label_video_splitter.py

import os
from shutil import copyfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from lipreading.config import args

class LabelVideoSplitter:
    def __init__(self):
        self.processed_mp4_folder = os.path.join(args["DATA_DIRECTORY"], 'processed_mp4')
        self.output_folder = os.path.join(args["DATA_DIRECTORY"], 'split_output')
        self.max_chars = args["MAIN_REQ_INPUT_LENGTH"]  # Character limit from config.py
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
    def process_label_file(self, label_file, input_video):
        """
        Splits label and video files based on word splits if text exceeds max_chars.
        """
        # Read the label file
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract the text part and word info
        text_line = lines[0].strip().replace("Text: ", "")
        words_info = lines[3:]  # Words start from line 4 onward (index 3)

        # Skip the header line "WORD START END" if present
        words_info = [line for line in words_info if line.strip() and not line.startswith("WORD")]

        # **Step 1**: Check if the text exceeds max_chars
        if len(text_line) <= self.max_chars:
            print(f"Label file {label_file} has {len(text_line)} characters. No split needed.")
            
            # Copy the original video and label to the output folder
            base_name = os.path.splitext(os.path.basename(label_file))[0]
            output_label_path = os.path.join(self.output_folder, f"{base_name}.txt")
            output_video_path = os.path.join(self.output_folder, f"{base_name}.mp4")
            
            # Copy label and video
            copyfile(label_file, output_label_path)
            copyfile(input_video, output_video_path)
            
            print(f"Copied original {output_video_path} and {output_label_path} to {self.output_folder}")
            return

        # **Step 2**: Split based on the number of words
        total_words = len(words_info)
        
        # Determine split point
        split_index = total_words // 2 + (total_words % 2)

        # First part (from the start to split_index)
        part1_labels = words_info[:split_index]
        part1_text = " ".join([l.split()[0] for l in part1_labels])
        part1_start_time = float(part1_labels[0].split()[1])
        part1_end_time = float(part1_labels[-1].split()[2])

        # Second part (from split_index to the end)
        part2_labels = words_info[split_index:]
        part2_text = " ".join([l.split()[0] for l in part2_labels])

        # Get the start time for part 2
        part2_start_time_original = float(part2_labels[0].split()[1])
        part2_end_time = float(part2_labels[-1].split()[2])

        # **Step 3**: Write new label files for part 1 and part 2 in the new folder
        base_name = os.path.splitext(os.path.basename(label_file))[0]

        # Write Part 1 label
        part1_label_file = os.path.join(self.output_folder, f"{base_name}_part_1.txt")
        with open(part1_label_file, 'w', encoding='utf-8') as f:
            f.write(f"Text: {part1_text}\n")
            f.write("Conf: 1\n\n")
            f.write("WORD START END\n")
            f.writelines(part1_labels)
            f.write("\n")

        # Adjust timings for part 2 to start from 0.0s
        new_part2_labels = []
        for label in part2_labels:
            word, start, end = label.strip().split()
            new_start = float(start) - part2_start_time_original  # Adjust start time
            new_end = float(end) - part2_start_time_original  # Adjust end time
            new_part2_labels.append(f"{word} {new_start:.2f} {new_end:.2f}\n")

        # Write Part 2 label
        part2_label_file = os.path.join(self.output_folder, f"{base_name}_part_2.txt")
        with open(part2_label_file, 'w', encoding='utf-8') as f:
            f.write(f"Text: {part2_text}\n")
            f.write("Conf: 1\n\n")
            f.write("WORD START END\n")
            f.writelines(new_part2_labels)

        # **Step 4**: Crop and save the videos for each part
        part1_video_file = os.path.join(self.output_folder, f"{base_name}_part_1.mp4")
        part2_video_file = os.path.join(self.output_folder, f"{base_name}_part_2.mp4")

        with VideoFileClip(input_video) as video:
            # Save Part 1: from the start to the start time of part 2
            part1_clip = video.subclip(part1_start_time, part2_start_time_original)
            part1_clip.write_videofile(part1_video_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            
            # Save Part 2: from the start time of part 2 to the end
            part2_clip = video.subclip(part2_start_time_original, part2_end_time)
            part2_clip.write_videofile(part2_video_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)

        print(f"Processed: {part1_video_file} and {part2_video_file} with corresponding label files.")
    
    def process_all_videos_labels(self):
        """
        Processes all video-label pairs and saves them to the split_output folder.
        """
        splitter = LabelVideoSplitter()
        video_files = sorted([f for f in os.listdir(splitter.processed_mp4_folder) if f.endswith('.mp4')])
        label_files = sorted([f for f in os.listdir(splitter.processed_mp4_folder) if f.endswith('.txt')])

        for video_file, label_file in zip(video_files, label_files):
            video_path = os.path.join(splitter.processed_mp4_folder, video_file)
            label_path = os.path.join(splitter.processed_mp4_folder, label_file)
            splitter.process_label_file(label_path, video_path)
