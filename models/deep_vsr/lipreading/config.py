# config.py

import os
from pathlib import Path

args = dict()

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
args["CODE_DIRECTORY"] = str(PROJECT_ROOT)  # Project root

# Allow overriding the data directory via environment variable while defaulting to ./data
data_root_env = os.getenv("DATA_DIRECTORY")
data_root = Path(data_root_env) if data_root_env else PROJECT_ROOT / "data"
args["DATA_DIRECTORY"] = str(data_root)
args["DEMO_DIRECTORY"] = str(PROJECT_ROOT / "final")
args["PRETRAINED_MODEL_FILE"] = str(PROJECT_ROOT / "models" / "video-only.pt")
args["TRAINED_MODEL_FILE"] = str(PROJECT_ROOT / "models" / "trained_model.pt")
args["TRAINED_LM_FILE"] = str(PROJECT_ROOT / "models" / "language_model.pt")
args["TRAINED_FRONTEND_FILE"] = str(PROJECT_ROOT / "models" / "visual_frontend.pt")

# Data
args["PRETRAIN_VAL_SPLIT"] = 0.01  # Validation set size fraction during pretraining
args["NUM_WORKERS"] = 4  # DataLoader num_workers argument
args["PRETRAIN_NUM_WORDS"] = 1  # Number of words limit in current curriculum learning iteration
args["MAIN_REQ_INPUT_LENGTH"] = 100  # Minimum input length while training

args["CHAR_TO_INDEX"] = {
    " ": 1, "'": 22, "1": 30, "0": 29, "3": 37, "2": 32, "5": 34, "4": 38, "7": 36, "6": 35, "9": 31, "8": 33,
    "A": 5, "Ă": 40, "Â": 41, "B": 20, "C": 17, "D": 12, "Đ": 46, "E": 2, "Ê": 42, "F": 19, "G": 16, "H": 9, 
    "I": 6, "K": 24, "L": 11, "M": 18, "N": 7, "O": 4, "Ô": 43, "Ơ": 44, "P": 21, "Q": 27, "R": 10, "S": 8, 
    "T": 3, "U": 13, "Ư": 45, "V": 23, "X": 26, "Y": 14, "Z": 28,
    "Á": 81, "À": 82, "Ả": 83, "Ã": 84, "Ạ": 85, "Ắ": 86, "Ằ": 87, "Ẳ": 88, "Ẵ": 89, "Ặ": 90, "Ấ": 91, "Ầ": 92, 
    "Ẩ": 93, "Ẫ": 94, "Ậ": 95, "É": 96, "È": 97, "Ẻ": 98, "Ẽ": 99, "Ẹ": 100, "Ế": 101, "Ề": 102, "Ể": 103, "Ễ": 104,
    "Ệ": 105, "Ó": 106, "Ò": 107, "Ỏ": 108, "Õ": 109, "Ọ": 110, "Ố": 111, "Ồ": 112, "Ổ": 113, "Ỗ": 114, "Ộ": 115,
    "Ớ": 116, "Ờ": 117, "Ở": 118, "Ỡ": 119, "Ợ": 120, "Ú": 121, "Ù": 122, "Ủ": 123, "Ũ": 124, "Ụ": 125, "Ứ": 126, 
    "Ừ": 127, "Ử": 128, "Ữ": 129, "Ự": 130, "Í": 131, "Ì": 132, "Ỉ": 133, "Ĩ": 134, "Ị": 135, "Ý": 136, "Ỳ": 137, 
    "Ỷ": 138, "Ỹ": 139, "Ỵ": 140, "<EOS>": 39,
    ".": 141, ",": 142, "?": 143, "!": 144, ":": 145, ";": 146, "-": 147, "\"": 148, "(": 149, ")": 150, "W": 151, "J": 152
}

args["INDEX_TO_CHAR"] = {v: k for k, v in args["CHAR_TO_INDEX"].items()}

# Preprocessing
args["VIDEO_FPS"] = 25  # Frame rate of the video clips
args["ROI_SIZE"] = 112  # Height and width of input greyscale lip region patch
args["NORMALIZATION_MEAN"] = 0.4161  # Mean value for normalization of greyscale lip region patch
args["NORMALIZATION_STD"] = 0.1688  # Standard deviation value for normalization of greyscale lip region patch

# Training
args["SEED"] = 19220297  # Seed for random number generators
args["BATCH_SIZE"] = 16  # Minibatch size
args["STEP_SIZE"] = 100  # Number of samples in one step (virtual epoch)
args["NUM_STEPS"] = 1600  # Maximum number of steps to train for (early stopping is used)
args["SAVE_FREQUENCY"] = 100  # Saving the model weights and loss/metric plots after every these many steps
# Early Stopping Parameters
args["EARLY_STOPPING_PATIENCE"] = 5  # Number of epochs with no improvement after which training will be stopped
args["EARLY_STOPPING_MIN_DELTA"] = 0.001  # Minimum change in the monitored quantity to qualify as an improvement

# Optimizer and Scheduler
args["INIT_LR"] = 1e-4  # Initial learning rate for scheduler
args["FINAL_LR"] = 1e-6  # Final learning rate for scheduler
args["LR_SCHEDULER_FACTOR"] = 0.5  # Learning rate decrease factor for scheduler
args["LR_SCHEDULER_WAIT"] = 25  # Number of steps to wait to lower learning rate
args["LR_SCHEDULER_THRESH"] = 0.001  # Threshold to check plateau-ing of WER
args["MOMENTUM1"] = 0.9  # Optimizer momentum 1 value
args["MOMENTUM2"] = 0.999  # Optimizer momentum 2 value

# Model
args["NUM_CLASSES"] = len(args["CHAR_TO_INDEX"])  # Number of output characters

# Transformer Architecture
args["PE_MAX_LENGTH"] = 2500  # Length up to which we calculate positional encodings
args["TX_NUM_FEATURES"] = 512  # Transformer input feature size
args["TX_ATTENTION_HEADS"] = 8  # Number of attention heads in multihead attention layer
args["TX_NUM_LAYERS"] = 6  # Number of Transformer Encoder blocks in the stack
args["TX_FEEDFORWARD_DIM"] = 2048  # Hidden layer size in feedforward network of transformer
args["TX_DROPOUT"] = 0.1  # Dropout probability in the transformer

# Beam Search
args["BEAM_WIDTH"] = 100  # Beam width
args["LM_WEIGHT_ALPHA"] = 0.5  # Weight of language model probability in shallow fusion beam scoring
args["LENGTH_PENALTY_BETA"] = 0.1  # Length penalty exponent hyperparameter
args["THRESH_PROBABILITY"] = 0.0001  # Threshold probability in beam search algorithm
args["USE_LM"] = False  # Whether to use language model for decoding

# Testing
args["TEST_DEMO_DECODING"] = "greedy"  # Test/demo decoding type - "greedy" or "search"

if __name__ == "__main__":
    for key, value in args.items():
        print(f"{key} : {value}")
