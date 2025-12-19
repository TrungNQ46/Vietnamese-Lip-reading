# Video-Only Lip Reading with Curriculum Learning

End-to-end pipeline that converts word-level annotated MP4 clips into training-ready visual features and trains a transformer-based lip-reading model using curriculum learning. The single entry point remains `python pipeline.py`.

## Requirements
- Python 3.9+ with `pip`
- FFmpeg available on the system path (required by MoviePy/OpenCV)
- NVIDIA GPU with CUDA 11.3+ and drivers for faster preprocessing and training (CPU works but is slow)
- Optional: Docker 24+ and Docker Compose v2 if you prefer containers

## Project Structure
- `lipreading/` – source package
  - `config.py` – runtime paths and hyperparameters
  - `pipeline.py` – orchestrates the full pipeline
  - `data_processing/` – CSV-to-TXT conversion, label/video splitting, split file creation, preprocessing
  - `datasets/` – dataset classes and collate helpers
  - `models/` – model definitions (VideoNet, visual frontend, language model)
  - `training/` – curriculum trainer
  - `utils/` – decoding, metrics, preprocessing helpers
- `data/` – data root (raw CSV/MP4 pairs, split files, preprocessed features)
- `models/` – place pretrained and finetuned `.pt` weights here
- `checkpoints/` – created during training (models and plots)
- `final/` – archived demo outputs
- `docker-compose.yml` and `Dockerfile` – containerized runner

## Data Layout (default `data/`)
- Place raw pairs in `data/processed_csv_final/`:
  - `sample.mp4`
  - `sample.csv` with columns `Word,Start,End` (in seconds; basename matches the MP4)
- Converter writes TXT labels to `data/processed_mp4/`; keep the paired MP4s there too.
- Splitter writes to `data/split_output/`; preprocessing later moves this folder to `data/main/split_output/`.
- Split lists are written to `data/pretrain.txt`, `train.txt`, `val.txt`, `test.txt`, and `preval.txt`.
- Preprocessed samples end up under `data/main/split_output/<name>.mp4/.txt/.png/.npy`.

## Environment Configuration
- `DATA_DIRECTORY` (optional): override the data root; defaults to `<repo>/data`.
- Example `.env` (works for shell and Docker Compose):
  ```
  DATA_DIRECTORY=D:\video_only\data
  ```
  Bash/macOS: `export DATA_DIRECTORY=$PWD/data`
  
  PowerShell: `$Env:DATA_DIRECTORY="$PWD\\data"`

## Local Development Setup
1) Create and activate a virtual environment:
   ```
   python -m venv .venv
   .\.venv\Scripts\activate     # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```
2) Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3) Prepare folders (create if missing):
   ```
   mkdir data data/processed_csv_final data/processed_mp4 data/split_output models checkpoints
   ```
4) Add weights to `models/`:
   - `visual_frontend.pt` (required for preprocessing)
   - `video-only.pt` (optional warm-start)
   - `language_model.pt` (optional for beam search)
   - `trained_model.pt` (latest checkpoint if resuming)
5) Run the pipeline:
   ```
   python pipeline.py
   ```

## Docker / Production Run
- Build and run with GPU passthrough (requires `nvidia-container-toolkit`):
  ```
  docker compose up --build
  ```
- Host folders `./data` -> `/app/data` and `./models` -> `/app/models` are mounted by default. Set `DATA_DIRECTORY` in a `.env` file if you need a different host path.
- Ad-hoc run without staying attached:
  ```
  docker compose run --rm --build --gpus all lipreading_pipeline python3 pipeline.py
  ```

## Pipeline Stages (executed by `pipeline.py`)
1) Convert CSV labels to TXT transcripts in `processed_mp4/`.
2) Split long utterances (longer than `MAIN_REQ_INPUT_LENGTH`) into halves and copy paired MP4/TXT into `split_output/`.
3) Create `pretrain/train/val/test` split lists.
4) Move `split_output/` into `data/main/` and preprocess MP4s into ROI PNGs and `.npy` visual features; generate `preval.txt`.
5) Train VideoNet with curriculum word counts `[1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37]`, auto-halving batch size on OOM, and early stopping when WER plateaus; checkpoints and plots land in `checkpoints/`.

## Key Configuration (`lipreading/config.py`)
- Paths: `DATA_DIRECTORY`, `DEMO_DIRECTORY`, `PRETRAINED_MODEL_FILE`, `TRAINED_MODEL_FILE`, `TRAINED_LM_FILE`, `TRAINED_FRONTEND_FILE`
- Data: `MAIN_REQ_INPUT_LENGTH`, `PRETRAIN_VAL_SPLIT`, `NUM_WORKERS`, `CHAR_TO_INDEX`
- Training: `BATCH_SIZE`, `NUM_STEPS`, `SAVE_FREQUENCY`, `INIT_LR`–`FINAL_LR`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_MIN_DELTA`
- Model: `TX_NUM_FEATURES`, `TX_ATTENTION_HEADS`, `TX_NUM_LAYERS`, `TX_FEEDFORWARD_DIM`, `TX_DROPOUT`, `PE_MAX_LENGTH`, `NUM_CLASSES`
- Curriculum list is defined inside `ModelTrainer.train_model()` (`lipreading/training/model_trainer.py`)

## Outputs
- Lists: `data/pretrain.txt`, `train.txt`, `val.txt`, `test.txt`, `preval.txt`
- Preprocessed samples: `data/main/split_output/<name>.mp4/.txt/.png/.npy`
- Checkpoints: `checkpoints/models/wordcount_<k>_step_<n>_wer_<x>.pt`
- Plots: `checkpoints/plots/wordcount_<k>_step_<n>_loss.png` and `_wer.png`

## Troubleshooting
- GPU not visible in Docker: ensure `--gpus all`, `nvidia-smi` works inside the container, and `nvidia-container-toolkit` is installed
- FFmpeg errors: confirm `ffmpeg` is on PATH and accessible inside Docker
- Missing weights: verify required `.pt` files exist in `models/` before preprocessing or training
- MP4/TXT mismatch: each basename in `processed_mp4/` needs both `.mp4` and `.txt`; the splitter zips sorted lists
- OOM during training: batch size auto-halves; you can also lower `BATCH_SIZE` in `config.py` or shorten the curriculum list
