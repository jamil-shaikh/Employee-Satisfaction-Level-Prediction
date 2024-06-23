# Employee Satisfaction Level Prediction

This project aims to predict employee satisfaction levels based on various features using machine learning techniques. The project involves data preprocessing, feature engineering, data visualization, model training, and optimization.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Deployment](#deployment)

## Overview

**Input:** Employee details (last evaluation score, number of projects, average monthly hours, time spent in the company, left, average hours per project, high_satisfaction).

**Output:** Predicted satisfaction level. 

**Web App:** Accessible via a basic Streamlit interface.

## Dataset

The dataset used contains features such as employee evaluation scores, project counts, working hours, and more.

## Features

- **Data Loading:** Load the dataset using pandas.
- **Data Preprocessing:** Correct column names, define features and target, and handle categorical and numerical columns.
- **Feature Engineering:** Create new features to enhance model performance.
- **Data Visualization:** Visualize numerical and categorical features using seaborn and matplotlib.
- **Model Training:** Train a machine learning model using a pipeline with preprocessing and a Random Forest regressor.
- **Model Optimization:** Optimize the model using Grid Search CV.
- **Model Saving:** Save the best model using joblib.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt

## Usage

Run the main script for training the models and making predictions:
python Regression.py


To run the Streamlit app:
streamlit run app.py

## Modeling

The modeling process includes:

    Defining a preprocessing pipeline for numerical and categorical features.
    Using a Random Forest regressor for the prediction task.
    Optimizing the model hyperparameters using Grid Search CV.

## Evaluation

The model is evaluated using:

    Mean Squared Error (MSE)
    R-squared (R²)
Need to add more

## Results

The optimized model's performance is evaluated, and the feature importance is visualized to understand the significant predictors of employee satisfaction.

## Deployment

The model is deployed using Streamlit. To access the web app, run:
streamlit run app.py
