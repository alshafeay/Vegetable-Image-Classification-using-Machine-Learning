# 🥕 Vegetable Image Classification using PCA & Random Forest 🤖

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20OpenCV%20%7C%20Kaggle-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains a machine learning project for classifying vegetable images into multiple categories. The notebook demonstrates a complete computer vision workflow, including data acquisition via the Kaggle API, image preprocessing with OpenCV, dimensionality reduction using Principal Component Analysis (PCA), and classification with a Random Forest model.

## 🗺️ Table of Contents

- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Methodology](#-methodology)
- [Results](#-results)
- [License](#-license)
- [Contact](#-contact)

## 📖 About The Project

In the age of digital information, the proliferation of fake news poses a significant challenge. This project addresses this problem by building a robust classification system to distinguish between real and fake news articles. It employs a comprehensive NLP pipeline to process and clean the text data, which is then converted into numerical features using TF-IDF. Finally, two supervised learning models are trained and evaluated to determine the most effective approach for this classification task.

## ✨ Key Features

- **🤖 Automated Data Acquisition:** Uses the Kaggle API to download and extract the dataset directly within the notebook.
- **🖼️ Image Preprocessing:** Leverages OpenCV to resize images to a uniform dimension for consistent feature vector length.
- **🔬 Dimensionality Reduction:** Applies Principal Component Analysis (PCA) to reduce the feature space, speeding up training.
- **⚖️ Feature Scaling:** Standardizes pixel values to ensure the model performs optimally.
- **🌳 Machine Learning Model:** Implements a `RandomForestClassifier`, a powerful ensemble model, for the classification task.
- **📊 Performance Evaluation:** Measures the model's effectiveness using the accuracy score on a held-out test set.

## 📊 Dataset

The project utilizes the "Vegetable Image Dataset" available on Kaggle.

- **Source:** [Kaggle Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Description:** The dataset contains thousands of 224x224 RGB images of vegetables, organized into training, testing, and validation sets. This project uses the images in the `train` directory.
- **Vegetable Categories (15 total):** Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish, Tomato.

## 💻 Tech Stack

- **Language:** Python 3
- **Core Libraries:**
  - **Data Manipulation:** `pandas`, `numpy`
  - **Machine Learning:** `scikit-learn`
  - **NLP:** `nltk`, `spacy`
  - **Visualization:** `matplotlib`, `seaborn`, `wordcloud`

## 📁 Project Structure
~~~
.
├── Vegetable_Image.ipynb       # Main Jupyter Notebook with all the code
└── vegetable_dataset/          # Folder created after unzipping
└── Vegetable Images/
├── train/              # Training images used in the notebook
├── test/
└── validation/
~~~
## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

- Python 3.7+
- A Kaggle account and an API key (`kaggle.json`)

### Installation

1.  **Clone the Repository:**
    ```sh
    git clone [https://github.com/your-username/vegetable-image-classification.git](https://github.com/your-username/vegetable-image-classification.git)
    cd vegetable-image-classification
    ```

2.  **Set up Kaggle API Key:**
    - Download your `kaggle.json` API token from your Kaggle account page.
    - Place the `kaggle.json` file in the root directory of this project. The notebook will automatically move it to the correct location (`~/.kaggle/`).

3.  **Install Required Packages:**
    ```sh
    pip install numpy opencv-python scikit-learn kaggle zipfile36 jupyter
    ```

4.  **Run the Notebook:**
    Launch Jupyter Notebook and open `Vegetable_Image.ipynb`. The notebook will handle the dataset download and extraction automatically.
    ```sh
    jupyter notebook
    ```

## ⚙️ Methodology

1.  **Data Acquisition:** The Kaggle dataset is downloaded and extracted using shell commands within the notebook.
2.  **Image Loading & Preprocessing:**
    - Images are loaded from each of the 15 category subdirectories.
    - Each image is resized to `32x32` pixels to create uniform feature vectors and reduce computational load.
    - The 3D image arrays (32x32x3) are flattened into 1D vectors of length 3072.
3.  **Data Preparation:**
    - The image vectors and their corresponding labels are stored in lists and then converted to NumPy arrays.
    - `LabelEncoder` is used to convert the string labels (e.g., "Carrot") into integers.
    - The data is split into an 80% training set and a 20% test set.
4.  **Feature Scaling & Reduction:**
    - `StandardScaler` is applied to standardize the feature values (pixel intensities).
    - **PCA** is fitted on the training data to reduce the dimensionality from 3072 to the top 100 principal components. Both training and test sets are then transformed using this PCA model.
5.  **Model Training & Evaluation:**
    - A `RandomForestClassifier` with 100 estimators is trained on the reduced-dimension training data.
    - The trained model is used to make predictions on the transformed test data.
    - The `accuracy_score` is calculated to evaluate the model's performance.

## 📈 Results

The Random Forest Classifier, trained on the 100 principal components of the image data, achieved the following result on the test set:

- **Model Accuracy:** **85.23%**

This demonstrates that a classic machine learning approach with effective dimensionality reduction can achieve high accuracy on a complex image classification task without requiring a deep learning architecture.

## 📄 License

This project is distributed under the MIT License.

## ✉️ Contact

Ahmed Alshafeay - [https://www.linkedin.com/in/ahmed-alshafeay-71b8b9300]
