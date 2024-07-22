# Potato--Plant-Disease-Classification


## Project Overview

This project aims to classify plant diseases using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The dataset used for training and testing the model is the PlantVillage dataset, which contains images of potato leaves categorized into three classes: 'Potato___Early_blight', 'Potato___Late_blight', and 'Potato___healthy'.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is the PlantVillage dataset, which is a collection of images of potato leaves. The images are categorized into three classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

The dataset is loaded using TensorFlow's `image_dataset_from_directory` function, which allows for easy preprocessing and batching of the images.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras. The architecture includes the following layers:
- Resizing and Rescaling layers for preprocessing
- Data Augmentation layers for improving generalization
- Multiple Conv2D and MaxPooling2D layers for feature extraction
- Flatten and Dense layers for classification


The dataset is split into training, validation, and test sets using an 80-10-10 split. The training and validation datasets are cached, shuffled, and prefetched to improve training performance.

## Evaluation

The model is evaluated on the test dataset after training. The evaluation metrics include loss and accuracy. The model's performance is visualized using plots of training and validation accuracy and loss over epochs.

## Results

The model achieves an accuracy of approximately 89.84% on the test dataset. The training and validation accuracy and loss are plotted to visualize the model's performance over epochs.

## Usage

To use this project, follow these steps:
1. Clone the repository from GitHub.
2. Install the required dependencies.
3. Run the Jupyter notebook to train and evaluate the model.
4. Save the trained model for future use.

## Dependencies

The project requires the following dependencies:
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Jupyter Notebook

## Acknowledgements

This project is based on the PlantVillage dataset and uses TensorFlow and Keras for building and training the model. Special thanks to the creators of the PlantVillage dataset and the developers of TensorFlow and Keras for their excellent tools and resources.
