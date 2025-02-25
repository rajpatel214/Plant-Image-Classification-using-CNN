# Plant-Image-Classification-using-CNN

## Overview
This deep learning project aims to classify plant images using a Convolutional Neural Network (CNN). The dataset contains multiple categories of plant images. The model is built using TensorFlow and Keras, achieving an accuracy of **71.91%**. This project demonstrates how deep learning techniques can be applied to image classification tasks effectively.

## Dataset
- **Source:** The dataset is sourced from Kaggle: [Avikumart Image Classification Dataset](https://www.kaggle.com/datasets/avikumart/imageclassificationdataset).
- **Structure:** The dataset consists of multiple folders, each representing a plant category.
- **Preprocessing:**
  - The dataset is extracted and stored in a structured directory format.
  - The number of images in each category is counted for analysis.
  - The dataset is normalized and augmented using `ImageDataGenerator` to enhance model generalization.

## Project Workflow
### Step 1: Data Loading & Preprocessing
- The dataset is downloaded from Kaggle and extracted using Python's `zipfile` module.
- The directory structure is analyzed to count the number of images per class.
- TensorFlow's `ImageDataGenerator` is used for:
  - Rescaling pixel values to [0,1] range.
  - Splitting the dataset into **training (80%)** and **validation (20%)** sets.
  - Applying real-time data augmentation techniques like rotation and zoom.

### Step 2: Model Architecture
The CNN model follows a structured approach with:
- **Convolutional Layers:**
  - Four sets of Conv2D layers with `ReLU` activation.
  - Kernel initialization using `HeNormal` for better weight distribution.
  - `l2` regularization to reduce overfitting.
- **Batch Normalization:**
  - Used after convolutional layers to stabilize training.
- **Max Pooling Layers:**
  - Used after each convolution block to reduce spatial dimensions.
- **Fully Connected Layers:**
  - `Flatten` layer to convert 2D feature maps into a 1D vector.
  - Dense layers with `ReLU` activation.
  - Dropout layer (`0.5`) to prevent overfitting.
  - Final output layer with `softmax` activation for multi-class classification.

### Step 3: Training & Evaluation
- **Loss Function:** `categorical_crossentropy` (since it's a multi-class classification problem).
- **Optimizer:** `Adam` optimizer for adaptive learning.
- **Metrics:** Accuracy is used as the evaluation metric.
- **Early Stopping:** Implemented to halt training if the validation accuracy stops improving.
- **Training Process:**
  - The model is trained for **20 epochs**.
  - Uses a batch size of **32**.
  - Tracks both training and validation accuracy/loss.

### Step 4: Results & Performance Evaluation
- The final model achieves a test accuracy of **71.91%**.
- The performance is visualized using matplotlib:
