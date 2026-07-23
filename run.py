import os
import subprocess
import sys

import config


def _ensure_dataset():
    if not os.path.exists(config.DATASET_PATH):
        print("• Dataset missing — generating sample data...")
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
    else:
        print("• Dataset found.")


def _ensure_model():
    if not os.path.exists(config.MODEL_PATH):
        print("• Model missing — training...")
        subprocess.run(
            [sys.executable, os.path.join("models", "train_model.py")], check=True
        )
    else:
        print("• Trained model found.")


def main():
    _ensure_dataset()
    _ensure_model()
    print(f"• Starting Flask server on http://127.0.0.1:{config.PORT} ...")
    # Import here so the dataset/model checks run before app import side-effects.
    from app import app

    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
