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










