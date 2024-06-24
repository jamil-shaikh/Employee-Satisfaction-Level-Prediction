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
- [Link](#link)

## Overview
Input: Employee details (last evaluation score, number of projects, average monthly hours, time spent in the company, left, average hours per project, high_satisfaction).

Output: Predicted satisfaction level.

Web App: Accessible via a basic Streamlit interface.

## Dataset
The dataset used contains features such as employee evaluation scores, project counts, working hours, and more. The dataset can be found [here](https://www.kaggle.com/datasets/liujiaqi/hr-comma-sepcsv).

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
```
## Usage

Run the main script for training the models and making predictions:
```bash
python main.py
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
- **MSE:** 0.030201456783105405
- **Root Mean Squared Error:** 0.17378566334167328
- **Mean Absolute Error:** 0.12811430347824132
- **R-squared:** 0.5114066931875252
- **Mean Absolute Percentage Error:** 30.27027507236356

The feature importance is visualized to understand the significant predictors of employee satisfaction.

## Deployment

The model is deployed using Streamlit. To access the web app, run:
```bash
streamlit run app.py
```
## Link to access the web app:
[here](https://employee-satisfaction-level-prediction-gu9rrhjoijitkx3zjduadf.streamlit.app/)

## Note
Use this csv file [here](https://github.com/jamil-shaikh/Employee-Satisfaction-Level-Prediction/blob/03ba5c2cad01d5c3c22de59b548dc4ee3fa64b71/HR_comma_sep.csv).

