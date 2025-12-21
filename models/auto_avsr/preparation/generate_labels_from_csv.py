#!/usr/bin/env python3
"""Generate repo-style label CSV from a CSV of (video_path, transcript).

Output lines have the format required by the repo:
  dataset_name,relative_path,input_length,token_id_sequence

Example:
  python preparation/generate_labels_from_csv.py \
    --input mylist.csv --root-dir /data/myroot --output ./labels/train.csv \
    --path-col path --transcript-col transcript --dataset-name myset

The script uses `datamodule.transforms.TextTransform` (SentencePiece) to encode
transcripts to token ids and `torchvision.io.read_video` to count frames.
"""
import argparse
import csv
import os
from typing import Optional

import torch
import torchvision

from datamodule.transforms import TextTransform


def find_video_path(root_dir: str, dataset_name: str, rel_or_abs_path: str) -> Optional[str]:
    # If path is absolute and exists, use it
    if os.path.isabs(rel_or_abs_path) and os.path.exists(rel_or_abs_path):
        return rel_or_abs_path
    # Try relative to root_dir/dataset_name
    candidate = os.path.join(root_dir, dataset_name, rel_or_abs_path)
    if os.path.exists(candidate):
        return candidate
    # Try relative to root_dir
    candidate2 = os.path.join(root_dir, rel_or_abs_path)
    if os.path.exists(candidate2):
        return candidate2
    # Try the raw path as-is
    if os.path.exists(rel_or_abs_path):
        return rel_or_abs_path
    return None


def count_frames(video_path: str) -> int:
    vid, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")
    return int(vid.shape[0])


def encode_transcript(tt: TextTransform, transcript: str) -> str:
    toks = tt.tokenize(transcript)
    toks = toks.cpu().numpy().tolist()
    return " ".join(map(str, toks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV file with video path and transcript columns")
    p.add_argument("--root-dir", required=True, help="Root directory where datasets (folders) live")
    p.add_argument("--output", required=True, help="Output label CSV (will be overwritten)")
    p.add_argument("--path-col", default="path", help="CSV column name for video path")
    p.add_argument("--transcript-col", default="transcript", help="CSV column name for transcript")
    p.add_argument("--dataset-name", default="myset", help="Dataset folder name under root_dir")
    p.add_argument("--rel-root", action="store_true", help="If set, write relative paths against root-dir/dataset_name")
    p.add_argument("--skip-missing", action="store_true", help="If set, skip missing video files instead of failing")
    args = p.parse_args()

    tt = TextTransform()

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)

        missing = 0
        for row in reader:
            if args.path_col not in row or args.transcript_col not in row:
                raise SystemExit(f"Input CSV must contain columns '{args.path_col}' and '{args.transcript_col}'")
            given_path = row[args.path_col].strip()
            transcript = row[args.transcript_col].strip()

            video_path = find_video_path(args.root_dir, args.dataset_name, given_path)
            if video_path is None:
                missing += 1
                if args.skip_missing:
                    continue
                else:
                    raise FileNotFoundError(f"Could not find video for {given_path}")

            # compute relative path (path used in labels should be relative to dataset folder)
            if args.rel_root:
                rel_path = os.path.relpath(video_path, start=os.path.join(args.root_dir, args.dataset_name))
            else:
                # keep the given path if it already looks relative, else compute relative to dataset folder
                if os.path.isabs(given_path):
                    rel_path = os.path.relpath(video_path, start=os.path.join(args.root_dir, args.dataset_name))
                else:
                    rel_path = given_path

            # count frames
            try:
                input_length = count_frames(video_path)
            except Exception as e:
                raise RuntimeError(f"Error reading video {video_path}: {e}")

            token_id_seq = encode_transcript(tt, transcript)
            writer.writerow([args.dataset_name, rel_path, str(int(input_length)), token_id_seq])

    if missing:
        print(f"Warning: {missing} missing videos (use --skip-missing to ignore)")
    print(f"Wrote labels to {args.output}")


if __name__ == "__main__":
    main()
