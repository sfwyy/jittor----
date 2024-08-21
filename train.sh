#!/bin/bash

# Set the dataset path and other parameters
DATASET_PATH="Dataset/"
MODEL_PATH="model/ViT-B-32.pkl"
SPLIT="A"

# Create directories if they don't exist
mkdir -p $DATASET_PATH
mkdir -p saved_models

# Execute the training script with arguments
python3 train.py --dataset_path $DATASET_PATH --model_path $MODEL_PATH --split $SPLIT

# Optional: After training, you can move or backup the trained model
# mv saved_models/test_data_strength16_10_epoch_*.pkl backup_directory/
