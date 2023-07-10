# Grapes Image Classification

This repository contains code for a grape image classification project using TensorFlow and VGG16 model.

## Overview

The code performs the following tasks:

- Downloads and preprocesses the grape images dataset
- Trains a VGG16 model on the training set
- Evaluates the model on the validation and test sets
- Generates classification reports and confusion matrix
- Makes predictions on new images

## Dependencies

The following dependencies are required to run the code:

- Python 3.7
- TensorFlow 2.x
- NumPy
- Pandas
- Seaborn
- Matplotlib
- PIL

## Usage

To run the code:

1. Clone the repository: `git clone https://github.com/majid0110/grapes-image-classification.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## File Structure

The repository has the following file structure:

grapes-image-classification/
├── main.py
├── model.py
├── dataset/
│ ├── grapes images/
│ │ ├── train/
│ │ ├── validation/
│ │ └── test/
│ └── grape_leaf.jpeg
└── README.md

- `main.py`: The main script that trains the model and makes predictions.
- `model.py`: Defines the model architecture and training configuration.
- `dataset/`: Contains the dataset folders and example test image.
- `README.md`: The README file you're currently reading.

## Results

Here are some results and visualizations obtained from running the code:

- Training and validation accuracy/loss curves
- Classification report on the validation set
- Confusion matrix
- Predicted class for a sample test image


