# Entry point wrapper to keep compatibility with Docker/CLI usage.
from lipreading.pipeline import main_pipeline


if __name__ == "__main__":
    main_pipeline()
