# Radar Signal Classification Project

## Overview
This project implements a deep learning classifier for radar signal modulation recognition using Convolutional Neural Networks (CNN).

## Project Structure
```
radar_signal_classification/
│
├── data/                  # Directory for datasets
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Neural network model definition
│   ├── train.py           # Training and evaluation logic
│   └── utils.py           # Visualization and metrics utilities
├── results/               # Training results
│   ├── models/            # Saved model checkpoints
│   ├── logs/              # Training logs and metrics
│   └── plots/             # Visualization outputs
├── main.py                # Main entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/radar_signal_classification.git
cd radar_signal_classification
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python main.py --dataset /path/to/your/dataset.hdf5 \
               --epochs 50 \
               --batch_size 64 \
               --cv_splits 5
```

### Parameters
- `--dataset`: Path to HDF5 dataset
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--cv_splits`: Number of cross-validation splits

## Results
Results are automatically saved in the `results/` directory:
- Model checkpoints in `results/models/`
- Training logs in `results/logs/`
- Visualization plots in `results/plots/`

## Model Architecture
- 1D Convolutional Neural Network
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Softmax output layer for multi-class classification

## Requirements
- Python 3.8+
- GPU recommended for faster training

## Contributing
Contributions are welcome. Please open an issue or submit a pull request.
