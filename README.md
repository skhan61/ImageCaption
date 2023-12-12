# Image Captioning Project

This project explores various deep learning architectures for the task of image captioning. It includes implementations of baseline CNN-LSTM models, as well as advanced architectures using Group Convolutional Neural Networks (GCNN) and Squeeze-Excitation (SE) attention mechanisms.

## Project Structure

- `notebooks/`: Jupyter notebooks for model implementation and evaluation.
- `src/`: Source code for the project, including model definitions and training scripts.
- `saved_models/`: Contains trained models ready for inference.

### Notebooks

1. `1_Baseline_CNN_LSTM.ipynb`: Introduction to the baseline CNN-LSTM model.
2. `2_GCNN_LSTM.ipynb`: Implementation of the GCNN-LSTM model.
3. `3_GCNN_SE_LSTM.ipynb`: Advanced GCNN model with SE attention.
4. `4_testing.ipynb`: Testing and comparison of the different models.

### Models

Model architectures are defined in `src/models/`. This includes the CNN and GCNN encoders, LSTM decoder, and the group CNN layer implementation.

### Training

The training process is scripted in `src/trainer/trainer.py`, detailing the training loop, optimization strategies, and logging.

### Saved Models

Pre-trained models are stored in `saved_models/`, which can be used for quick testing and evaluation.

## Source Code

The `src/` directory contains the source code for the image captioning project.

### Components

- `dataset/`: Dataset handling, including loading and preprocessing.
- `models/`: Core models including encoders and decoders for image captioning.
- `trainer/`: Training script with the training loop and optimization.
- `utils/`: Auxiliary utilities and helper functions.