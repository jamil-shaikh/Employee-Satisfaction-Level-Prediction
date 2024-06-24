# Employee Satisfaction Level Prediction

This project aims to predict employee satisfaction levels based on various features using machine learning techniques. Predicting employee satisfaction can help organizations improve workplace conditions and reduce turnover rates.

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
Input: Employee details (last evaluation score, number of projects, average monthly hours, time spent in the company, left, average hours per project, high_satisfaction).

Output: Predicted satisfaction level.

Web App: Accessible via a basic Streamlit interface.

## Dataset
The dataset used contains features such as employee evaluation scores, project counts, working hours, and more. The dataset can be found [here](https://github.com/jamil-shaikh/Employee-Satisfaction-Level-Prediction/blob/9051318502241627eeb92abd5f1eb0bb15e3f3e1/HR_comma_sep.csv).

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
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt



## Usage

Run the main script for training the models and making predictions:
```bash
python Regression.py
```

To run the Streamlit app:
```bash
streamlit run app.py
```

## Modeling

The modeling process includes:
- Defining a preprocessing pipeline for numerical and categorical features.
- Using a Random Forest regressor for the prediction task.
- Optimizing the model hyperparameters using Grid Search CV.

## Evaluation

The model is evaluated using:
- Mean Squared Error (MSE)
- R-squared (R²)
- Mean Absolute Error
- Root Mean Error
- Mean Absolute Percentage Error


## Results

The optimized model's performance is evaluated:
- **MSE:** Value
- **R²:** Value
- **MAE:** Value

The feature importance is visualized to understand the significant predictors of employee satisfaction.

## Deployment

The model is deployed using Streamlit. To access the web app, run:
```bash
streamlit run app.py
