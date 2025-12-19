# model_trainer.py

import os
import shutil
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lipreading.config import args
from lipreading.models.video_net import VideoNet
from lipreading.datasets.lrs2_dataset import LRS2Main
from lipreading.datasets.utils import collate_fn
from lipreading.utils.general import num_params, train, evaluate

class ModelTrainer:
    """
    Class to handle the training of the VideoNet model with curriculum learning.
    """

    def __init__(self):
        self.code_directory = args["CODE_DIRECTORY"]
        self.checkpoints_dir = os.path.join(self.code_directory, "checkpoints")
        self.models_dir = os.path.join(self.checkpoints_dir, "models")
        self.plots_dir = os.path.join(self.checkpoints_dir, "plots")
        self._setup_directories()

    def _setup_directories(self):
        """
        Sets up the checkpoints, models, and plots directories.
        If checkpoints directory exists, prompts to remove it.
        """
        if os.path.exists(self.checkpoints_dir):
            while True:
                ch = input("Continue and remove the 'checkpoints' directory? y/n: ").lower()
                if ch == "y":
                    try:
                        shutil.rmtree(self.checkpoints_dir)
                        print("Removed existing 'checkpoints' directory.")
                    except Exception as e:
                        print(f"Error removing 'checkpoints' directory: {e}")
                        sys.exit()
                    break
                elif ch == "n":
                    print("Exiting the pipeline.")
                    sys.exit()
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)
            print(f"Created directories: {self.models_dir}, {self.plots_dir}")
        except Exception as e:
            print(f"Error creating checkpoint directories: {e}")
            sys.exit()

    def train_model(self):
        """
        Trains the VideoNet model using curriculum learning.
        Iterates over predefined word counts and adjusts batch size upon OOM errors.
        """
        curriculum_word_counts = [1, 2, 3, 5, 7, 9, 13, 17, 21, 29, 37]
        initial_batch_size = args["BATCH_SIZE"]

        for word_count in curriculum_word_counts:
            print(f"\n=== Starting Curriculum Learning Iteration with {word_count} words ===")
            args["PRETRAIN_NUM_WORDS"] = word_count  # Update the number of words in config
            batch_size = initial_batch_size
            oom_encountered = False

            while True:
                try:
                    self._train_iteration(batch_size, word_count)
                    break  # Break the loop if training completes without OOM
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"Out Of Memory error encountered with batch size {batch_size}. Reducing batch size by half.")
                        torch.cuda.empty_cache()
                        batch_size = max(1, batch_size // 2)
                        oom_encountered = True
                        if batch_size == 1:
                            print("Minimum batch size reached. Cannot reduce further.")
                            break
                    else:
                        raise e

            if oom_encountered:
                print(f"Completed iteration with adjusted batch size {batch_size} due to OOM errors.")
            else:
                print(f"Completed iteration with batch size {batch_size}.")

    def _train_iteration(self, batch_size, word_count):
        """
        Trains the model for a single curriculum learning iteration.
        Implements early stopping based on validation WER flattening and learning rate schedule.
        """
        data_directory = args["DATA_DIRECTORY"]
        matplotlib.use("Agg")  # For environments without a display
        np.random.seed(args["SEED"])
        torch.manual_seed(args["SEED"])
        gpu_available = torch.cuda.is_available()
        device = torch.device("cuda" if gpu_available else "cpu")
        kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpu_available else {}
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load datasets with the specified number of words
        video_params = {"videoFPS": args["VIDEO_FPS"]}
        train_data = LRS2Main(
            "train",
            data_directory,
            args["MAIN_REQ_INPUT_LENGTH"],
            args["CHAR_TO_INDEX"],
            args["STEP_SIZE"],
            video_params,
            num_words=word_count  # Pass the number of words for curriculum learning
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            **kwargs
        )
        val_data = LRS2Main(
            "val",
            data_directory,
            args["MAIN_REQ_INPUT_LENGTH"],
            args["CHAR_TO_INDEX"],
            args["STEP_SIZE"],
            video_params,
            num_words=word_count  # Ensure validation data matches the training data
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            **kwargs
        )

        # Initialize model, optimizer, scheduler, and loss function
        model = VideoNet(
            args["TX_NUM_FEATURES"],
            args["TX_ATTENTION_HEADS"],
            args["TX_NUM_LAYERS"],
            args["PE_MAX_LENGTH"],
            args["TX_FEEDFORWARD_DIM"],
            args["TX_DROPOUT"],
            args["NUM_CLASSES"]
        )
        model.to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args["INIT_LR"],
            betas=(args["MOMENTUM1"], args["MOMENTUM2"])
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args["LR_SCHEDULER_FACTOR"],
            patience=args["LR_SCHEDULER_WAIT"],
            threshold=args["LR_SCHEDULER_THRESH"],
            threshold_mode="abs",
            min_lr=args["FINAL_LR"],
            verbose=True
        )
        loss_function = nn.CTCLoss(blank=0, zero_infinity=False)

        training_loss_curve = []
        validation_loss_curve = []
        training_wer_curve = []
        validation_wer_curve = []

        # Early stopping criteria
        patience = args.get("EARLY_STOPPING_PATIENCE", 5)
        min_delta = args.get("EARLY_STOPPING_MIN_DELTA", 0.001)
        patience_counter = 0
        best_val_wer = float('inf')
        lr_reached_min = False

        print(f"\nStarting training with batch size {batch_size} and {word_count} words per sample.")

        train_params = {
            "spaceIx": args["CHAR_TO_INDEX"].get(" ", 1),
            "eosIx": args["CHAR_TO_INDEX"].get("<EOS>", 39)
        }
        val_params = {
            "decodeScheme": "greedy",
            "spaceIx": args["CHAR_TO_INDEX"].get(" ", 1),
            "eosIx": args["CHAR_TO_INDEX"].get("<EOS>", 39)
        }

        for step in range(args["NUM_STEPS"]):
            # Train the model for one step
            training_loss, training_cer, training_wer = train(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
                train_params
            )
            training_loss_curve.append(training_loss)
            training_wer_curve.append(training_wer)

            # Evaluate the model on validation set
            validation_loss, validation_cer, validation_wer = evaluate(
                model,
                val_loader,
                loss_function,
                device,
                val_params
            )
            validation_loss_curve.append(validation_loss)
            validation_wer_curve.append(validation_wer)

            # Printing the stats after each step
            print(f"Step: {step+1:04d}/{args['NUM_STEPS']} || Tr.Loss: {training_loss:.6f}  Val.Loss: {validation_loss:.6f} || "
                  f"Tr.CER: {training_cer:.3f}  Val.CER: {validation_cer:.3f} || "
                  f"Tr.WER: {training_wer:.3f}  Val.WER: {validation_wer:.3f}")

            # Check for early stopping based on validation WER flattening
            if validation_wer + min_delta < best_val_wer:
                best_val_wer = validation_wer
                patience_counter = 0
            else:
                patience_counter += 1

            # Check if learning rate has reached the minimum
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= args["FINAL_LR"]:
                lr_reached_min = True

            # Make a scheduler step
            scheduler.step(validation_wer)

            # Saving the model weights and loss/metric plots after every few steps
            if ((step + 1) % args["SAVE_FREQUENCY"] == 0) or (step + 1 == args["NUM_STEPS"]):
                save_path = os.path.join(
                    self.models_dir,
                    f"wordcount_{word_count}_step_{step+1:04d}_wer_{validation_wer:.3f}.pt"
                )
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f"Saved model checkpoint to {save_path}")
                except Exception as e:
                    print(f"Error saving model checkpoint at step {step+1}: {e}")

                # Plot Loss Curves
                try:
                    plt.figure()
                    plt.title(f"Loss Curves (Word Count: {word_count})")
                    plt.xlabel("Step No.")
                    plt.ylabel("Loss value")
                    plt.plot(range(1, len(training_loss_curve) + 1), training_loss_curve, "blue", label="Train")
                    plt.plot(range(1, len(validation_loss_curve) + 1), validation_loss_curve, "red", label="Validation")
                    plt.legend()
                    loss_plot_path = os.path.join(self.plots_dir, f"wordcount_{word_count}_step_{step+1:04d}_loss.png")
                    plt.savefig(loss_plot_path)
                    plt.close()
                    print(f"Saved loss plot to {loss_plot_path}")
                except Exception as e:
                    print(f"Error plotting loss curves at step {step+1}: {e}")

                # Plot WER Curves
                try:
                    plt.figure()
                    plt.title(f"WER Curves (Word Count: {word_count})")
                    plt.xlabel("Step No.")
                    plt.ylabel("WER")
                    plt.plot(range(1, len(training_wer_curve) + 1), training_wer_curve, "blue", label="Train")
                    plt.plot(range(1, len(validation_wer_curve) + 1), validation_wer_curve, "red", label="Validation")
                    plt.legend()
                    wer_plot_path = os.path.join(self.plots_dir, f"wordcount_{word_count}_step_{step+1:04d}_wer.png")
                    plt.savefig(wer_plot_path)
                    plt.close()
                    print(f"Saved WER plot to {wer_plot_path}")
                except Exception as e:
                    print(f"Error plotting WER curves at step {step+1}: {e}")

            # Check for early stopping condition
            if patience_counter >= patience and lr_reached_min:
                print(f"Early stopping triggered at step {step+1} due to validation WER flattening.")
                break

        print(f"\nTraining for word count {word_count} completed.\n")
