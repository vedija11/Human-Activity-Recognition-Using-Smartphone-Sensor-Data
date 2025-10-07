# Human Activity Recognition Using Smartphone Sensor Data

This project implements a machine learning pipeline to classify six common human activities—**LAYING, SITTING, STANDING, WALKING, WALKING_UPSTAIRS, and WALKING_DOWNSTAIRS**—using accelerometer and gyroscope data from smartphones.

## Project Overview

The dataset contains **10,297 observations** with **564 sensor-derived features**. The goal was to design an efficient and accurate classification model while mitigating high dimensionality and evaluating multiple algorithms.

Key steps include:

* **Data Preprocessing:** Verified missing values, standardized numeric predictors, and removed redundant columns.
* **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** to reduce 564 features to 10 components (~64% variance retained).
* **Model Training & Evaluation:** Compared **k-Nearest Neighbors (kNN), Random Forest, SVM (RBF), and XGBoost** using a 70/30 train-test split with 5-fold cross-validation.
* **Best Model:** **Random Forest** achieved the highest accuracy (71%) and Cohen’s Kappa (0.651), providing balanced performance across activities.

## Insights & Results

* PCA effectively reduced feature redundancy and highlighted overlapping activity patterns (e.g., SITTING vs. STANDING).
* Random Forest handled non-linear interactions and noisy features better than distance- or margin-based classifiers.
* Confusion matrices revealed strong prediction for dynamic activities like WALKING_DOWNSTAIRS and challenges distinguishing static postures.

## Limitations & Future Work

* Accuracy is moderate (<75%) due to class imbalance and similarity between certain activities.
* Data was collected in a controlled environment; real-world deployment may require diverse data.
* Future work includes exploring **deep learning architectures (CNN/LSTM)**, collecting more diverse datasets, and applying techniques to address class imbalance.

## Tools & Technologies

* **Languages:** R
* **Libraries:** caret, xgboost, ggplot2, ggcorrplot, dplyr, tidyr
* **Techniques:** PCA, Random Forest, kNN, SVM, XGBoost, cross-validation
