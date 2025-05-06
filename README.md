# Sonar-Rock-vs-Mine-Classification

This Streamlit application uses machine learning to classify sonar signals as either rocks or mines based on their frequency patterns.

# Overview
The app uses a Logistic Regression model trained on the sonar dataset, which contains patterns obtained by bouncing sonar signals off metal cylinders (mines) and rocks under various conditions.

# Features

**Data Exploration:** Visualize the dataset, including class distribution and feature patterns

**Model Performance:** Evaluate the model's performance with metrics and visualizations

**Make Prediction:** Input your own sonar readings to classify objects as rocks or mines

**Interactive UI:** User-friendly interface with visualizations and easy navigation

# Installation

1.Clone this repository

2.Install the required packages:

pip install -r requirements.txt

3.Download the sonar dataset (sonar data.csv) and place it in the same directory as the app

# Usage

1.Run the Streamlit app:

streamlit run sonar_app.py

2.Open your web browser and go to the URL displayed in the terminal (usually http://localhost:8501)

Upload the sonar dataset or use the preloaded data

Explore the data, check model performance, and make predictions

# Requirements

Python 3.7+

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

# Deployment
You can deploy this app on Streamlit Cloud by following these steps:

Push your code to a GitHub repository

Go to Streamlit Cloud

Connect your GitHub account

Select the repository and branch

Set the main file path to sonar_classification_app.py

Deploy!

# About the Dataset
The dataset consists of 208 patterns obtained by bouncing sonar signals off a metal cylinder (mine) and rocks under various conditions. Each pattern is a set of 60 numbers in the range 0.0 to 1.0, representing the energy within particular frequency bands integrated over time.
