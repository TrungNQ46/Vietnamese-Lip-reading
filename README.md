# VLR Dataset Preprocessing

This repository contains the preprocessing code used to construct the **VLR (Vietnamese Lip-Reading)** dataset from publicly available online videos.

The notebooks here implement the full pipeline from raw videos to utterance-level mouth-region clips and label files used in the paper:

> **VLR: A Vietnamese Lip-Reading Benchmark and Deep Learning–Based Evaluation of State-of-the-Art Models** (under review)

The **dataset itself** is hosted separately on Zenodo:

- **VLR dataset (videos + labels):** https://zenodo.org/records/15899222

This repository only contains **code**, not the original videos.

---

## 1. Overview

The preprocessing pipeline consists of four main stages:

1. **Video collection (external step)**  
   - Download public videos from platforms such as YouTube (e.g., educational channels, talk shows, public talks).  
   - Organize them into folders by playlist/channel.

2. **ASR-based segmentation (`VideoToCsv.ipynb`)**  
   - Use Whisper to transcribe each video.  
   - Export sentence/segment-level time stamps and transcripts to CSV files.

3. **Mouth-region cropping (`cropvideo.ipynb`)**  
   - Use MediaPipe + OpenCV to detect faces and track them across frames.  
   - Crop the mouth (or full face) region for each utterance according to the CSV time stamps.  
   - Save utterance-level video clips and their labels.

4. **Exploratory data analysis (`EDA.ipynb`)**  
   - Inspect statistics (number of clips, duration, length distributions, etc.).  
   - Check for missing or problematic samples.

Additional notebooks (`draft*.ipynb`, `ss1.ipynb`) contain development versions and helper utilities used during pipeline design.

---

## 2. Repository structure

Typical layout of this repository:

```text
.
├─ EDA.ipynb              # Exploratory data analysis on processed data
├─ VideoToCsv.ipynb       # Whisper-based transcription & alignment
├─ cropvideo.ipynb        # Face/mouth detection and cropping
├─ draft.ipynb            # Prototype / scratch code (not required for main pipeline)
├─ draft2.ipynb
├─ draft3.ipynb
├─ ss1.ipynb              # Additional experiments / utilities
└─ README.md
