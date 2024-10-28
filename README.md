# Face Mask Detection with MobileNetV3 using pytorch and openCV

This repository contains code for training a face mask detection model using MobileNetV3. It includes modularized scripts and Jupyter notebooks for training, testing, and live inference.

## Repository Structure

```
FACE_MASK_MOBILENET_LARGE
├── artifacts                  # Directory to save trained model checkpoints
├── input                      # Directory for dataset
│   ├── train                  # Training dataset
│   ├── val                    # Validation dataset
│   └── test                   # Test dataset
├── notebooks                  # Jupyter notebooks for experiments and inference
│   ├── experiment_mobilenet_large.ipynb
│   └── live_inference_webcam.ipynb
├── src                        # Modular code for training, data loading, and inference
│   ├── dataloaders.py         # Data loading utilities
│   ├── engine.py              # Training and evaluation engine
│   ├── live_inference_single_face.py  # Script for single face inference
│   ├── live_inference_multiple_faces.py # Script for multiple face inference
│   ├── model_builder.py       # Model architecture definition
│   └── train.py               # Main script to train the model
└── requirements.txt           # Python dependencies
```

## Getting Started

### 1. Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FACE_MASK_MOBILENET_LARGE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Dataset Preparation
Ensure your dataset follows the structure inside the `input` directory:
- `train`: Contains training images, organized in subdirectories by class (e.g., `WithMask` and `WithoutMask`).
- `val`: Contains validation images, organized similarly to the training set.
- `test`: Contains test images, organized similarly to the training set.

### 3. Training the Model

The model training code is modularized in the `src` directory. To train the model, run the `train.py` script with the following command:

```bash
python src/train.py --lr 0.0001 --epochs 10 --batch_size 32 --seed 42 \
    --train_dir "input/train" --val_dir "input/val" --test_dir "input/test"
```

#### Command-Line Arguments
- `--lr`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--seed`: Random seed for reproducibility (default: 42)
- `--train_dir`: Path to the training dataset
- `--val_dir`: Path to the validation dataset
- `--test_dir`: Path to the test dataset

### 4. Running Experiments in Notebooks

If you prefer experimenting interactively, the `notebooks` directory contains Jupyter notebooks:
- `experiment_mobilenet_large.ipynb`: Contains code for training and evaluating the model.
- `live_inference_webcam.ipynb`: Contains code for performing live face mask detection using a webcam.

### 5. Inference Scripts
For live inference, use the following scripts in the `src` directory:
- **Single Face Inference**: `live_inference_single_face.py`
- **Multiple Faces Inference**: `live_inference_multiple_faces.py`

Run them with:
```bash
python src/live_inference_single_face.py
# or
python src/live_inference_multiple_faces.py
```


