# Colon Cancer Classification using Convolutional Neural Networks

This project implements a Deep Learning model to classify histopathological images of colon tissue into two categories: **Benign (healthy tissue)** and **Malignant (Adenocarcinoma)**. The model is built using TensorFlow and Keras.

## 🚀 Project Overview

The primary goal of this project is to develop a robust Convolutional Neural Network (CNN) capable of accurately identifying cancerous tissues from microscopic images. This automated approach can serve as a valuable tool to assist pathologists in diagnosing colon cancer, potentially leading to faster and more efficient diagnoses.

## 📊 Dataset

The model was trained on a public dataset of lung and colon cancer histopathological images. This project specifically utilizes the colon tissue images.

* **Source**: The dataset was sourced from a collection of microscopic images.
* **Classes**: The data is divided into two distinct classes:
    * `Colon_Adenocarcinoma` (Malignant)
    * `Colon_Benign_Tissue` (Benign)

### Data Samples

Here are some examples of the images used for training the model:

![Data Samples](images/download%20%281%29.png)

## ⚙️ Workflow & Preprocessing

The project follows a standard machine learning pipeline:

1.  **Environment Setup**: The project was developed in a Google Colab environment, leveraging GPU acceleration for training.
2.  **Data Loading**: Data was loaded and unzipped directly from a cloud source.
3.  **Data Augmentation & Preprocessing**: To prevent overfitting and increase the diversity of the training set, Keras's `ImageDataGenerator` was used to perform:
    * **Rescaling**: Pixel values were normalized to the `[0, 1]` range.
    * **Augmentation**: Random transformations like shearing, zooming, and horizontal flipping were applied to the training images.
4.  **Data Generators**: The dataset was split into training (80%) and validation (20%) sets. Three data generators (`train`, `validation`, `test`) were created to feed images to the model in batches of 32, with all images resized to `224x224` pixels.

## 🤖 Model Architecture

A Sequential CNN model was constructed with the following layers, designed to capture features of increasing complexity:

| Layer              | Configuration                      | Purpose                                            |
| ------------------ | ---------------------------------- | -------------------------------------------------- |
| **Conv2D Block 1** | 32 Filters (3x3), ReLU Activation  | Extracts basic features like edges and textures.   |
| `BatchNormalization` | -                                | Stabilizes and accelerates learning.               |
| `MaxPooling2D`     | (2x2) Pool Size                    | Reduces spatial dimensions (downsampling).         |
| **Conv2D Block 2** | 64 Filters (3x3), ReLU Activation  | Learns more complex patterns from initial features.|
| `BatchNormalization` | -                                |                                                    |
| `MaxPooling2D`     | (2x2) Pool Size                    |                                                    |
| **Conv2D Block 3** | 128 Filters (3x3), ReLU Activation | Captures higher-level, more abstract features.     |
| `BatchNormalization` | -                                |                                                    |
| `MaxPooling2D`     | (2x2) Pool Size                    |                                                    |
| **Classifier Head**|                                    |                                                    |
| `Flatten`          | -                                  | Converts 2D feature maps into a 1D vector.         |
| `Dense`            | 128 Neurons, ReLU Activation       | A fully connected layer for high-level reasoning.  |
| `Dropout`          | Rate = 0.5                         | Prevents overfitting by randomly dropping neurons. |
| `Dense` (Output)   | 2 Neurons, Softmax Activation      | Produces the final probability for each class.     |

## 📈 Training & Evaluation

The model was compiled with the `Adam` optimizer, `categorical_crossentropy` loss function, and `accuracy` as the evaluation metric. It was trained for 10 epochs.

* **Final Validation Accuracy**: **97.28%**

### Training Progress

![Training Progress](images/Screenshot%202025-02-20%20140050.png)

### Performance Evaluation

The model's performance on the validation set is summarized by the confusion matrix below, which shows excellent results, especially in correctly identifying all malignant cases presented.

![Confusion Matrix](images/download%20(2).png)

* **True Malignant Predictions**: 1000
* **True Benign Predictions**: 930
* **False Malignant Predictions (Type I Error)**: 70
* **False Benign Predictions (Type II Error)**: 0

## 🔬 Inference & Prediction Examples

A prediction function was created to test the model on new, unseen images. The results demonstrate the model's high confidence in its classifications.

#### **Example 1: Malignant Prediction (100.00% Confidence)**

![Malignant Prediction](images/Screenshot%202025-04-14%20153035.png)

![Malignant Prediction](images/Screenshot%202025-05-10%20150345.png)

#### **Example 2: Benign Prediction (99.95% Confidence)**

![Benign Prediction](images/Screenshot%202025-04-14%20153056.png)

![Benign Prediction](images/Screenshot%202025-05-10%20150436.png)

## 🛠️ Technologies Used

* **Python 3**
* **TensorFlow & Keras**: For building and training the neural network.
* **Scikit-learn**: For generating the confusion matrix and classification report.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For data visualization and plotting.
* **Google Colab**: As the development and training environment.






