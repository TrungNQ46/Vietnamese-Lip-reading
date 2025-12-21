import argparse
import os
import random
import shutil
import pandas as pd
from rich import print
from tqdm import tqdm
from os.path import join


def split_data(dir, ratio=0.2, seed=42):

    videos = [f[:-4] for f in os.listdir(dir) if f.endswith(".mp4")]

    random.seed(seed)
    random.shuffle(videos)
    split_data = int(len(videos) * ratio)

    test_videos = videos[:split_data]
    train_videos = videos[split_data:]

    return train_videos, test_videos


def turn_into_lrs2(data_dir, videos):
    vid_to_remove = []
    for video_name in tqdm(videos, desc="Converting data to LRS2 structure"):
        # video_name = video[:-4]
        if os.path.exists(join(data_dir, video_name + ".mp4")) and os.path.exists(
            join(data_dir, video_name + ".csv")
        ):
            df = pd.read_csv(join(data_dir, video_name + ".csv"))
            text = "Text:  " + " ".join(
                ["".join(e for e in x if e.isalnum()) for x in list(df["Word"])]
            )
            try:
                with open(join(data_dir, video_name + ".txt"), "w") as f:
                    f.write(text)
                    f.write("\n")
                    # write the df to the file
                    f.write("WORD START END ASDSCORE\n")
                    for i in range(len(df)):
                        f.write(
                            f"{''.join(e for e in df['Word'][i] if e.isalnum())} {df['Start'][i]} {df['End'][i]} {1}\n"
                        )

            except Exception as e:
                tqdm.write(f"Error: {e}")
                tqdm.write(f"Error in writing text file for {video_name}. Skipping...")
                vid_to_remove.append(video)
                try:
                    os.remove(join(data_dir, video_name + ".txt"))
                except:
                    pass
        else:
            tqdm.write(f"Text/Video file not found for {video_name}. Skipping...")
            vid_to_remove.append(video)

    videos = [x for x in videos if x not in vid_to_remove]
    return videos


# create fucntion arguments
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()
    return args


def get_videos(data_dir):
    train_videos = []
    test_videos = []
    val_videos = []

    train_path = join(data_dir, "split", "train.txt")
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            for line in f:
                tmp = line.strip()
                tmp = tmp.replace("\\", "/")
                train_videos.append(tmp[:-4])

    test_path = join(data_dir, "split", "test.txt")
    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            for line in f:
                tmp = line.strip()
                tmp = tmp.replace("\\", "/")
                test_videos.append(tmp[:-4])

    val_path = join(data_dir, "split", "val.txt")
    if os.path.exists(val_path):
        with open(val_path, "r") as f:
            for line in f:
                tmp = line.strip()
                tmp = tmp.replace("\\", "/")
                val_videos.append(tmp[:-4])

    return train_videos, test_videos, val_videos


if __name__ == "__main__":
    cfg = create_args()
    data_dir = cfg.data_dir

    train_videos, test_videos, val_videos = get_videos(data_dir)

    train_videos = turn_into_lrs2(data_dir, train_videos)
    test_videos = turn_into_lrs2(data_dir, test_videos)
    val_videos = turn_into_lrs2(data_dir, val_videos)
    os.makedirs(join(data_dir, "data_split"), exist_ok=True)
    with open(join(data_dir, "data_split", "pretrain.txt"), "w") as f:
        pass

    with open(join(data_dir, "data_split", "train.txt"), "w") as f:
        for video in train_videos:
            f.write(video + "\n")

    with open(join(data_dir, "data_split", "test.txt"), "w") as f:
        for video in test_videos:
            f.write(video + "\n")

    with open(join(data_dir, "data_split", "val.txt"), "w") as f:
        for video in val_videos:
            f.write(video + "\n")
