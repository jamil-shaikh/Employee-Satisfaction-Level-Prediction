#!/usr/bin/env python
# coding: utf-8

# In[31]:


'''  Employee Satisfaction Prediction Model Training

This notebook handles the training and evaluation of the Random Forest model for predicting employee satisfaction.

Data Loading and Preprocessing

We start by loading the dataset and performing the necessary preprocessing steps.'''


# In[32]:


import pandas as pd

# Load the dataset
file_path = "C:/Users/Jamil Shaikh/Desktop/HR_comma_sep.csv"
data = pd.read_csv(file_path)


# Define features and target
X = data.drop(columns=['satisfaction_level'])
y = data['satisfaction_level']


# In[33]:


## Feature Engineering

# We create new features to improve model performance.


# In[34]:


# Feature Engineering: Creating new features
data['average_monthly_hours_per_year'] = data['average_monthly_hours'] * 12
data['average_hours_per_project'] = data['average_monthly_hours'] / data['number_project']
data['high_satisfaction'] = (data['satisfaction_level'] > 0.7).astype(int)


# In[35]:


''''## Exploratory Data Analysis

We visualize the distributions of numerical and categorical features.'''


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

numerical_cols = ['last_evaluation', 'number_project', 'average_monthly_hours', 'time_spent_company', 'Work_accident', 'left', 'promotion_last_5years', 'average_monthly_hours_per_year', 'average_hours_per_project']
categorical_cols = ['department', 'salary', 'high_satisfaction']

# Distribution of numerical features
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# In[37]:


# Count plots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=data)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[38]:


# Correlation matrix for numerical features
plt.figure(figsize=(12, 8))
corr_matrix = data[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features')
plt.show()


# In[39]:


''''## Model Training

We train a Random Forest model using GridSearchCV to find the best parameters.'''


# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
import matplotlib.pyplot as plt



# Create new features for Model 2
data['average_hours_per_project'] = data['average_monthly_hours'] / data['number_project']
data['high_satisfaction'] = (data['satisfaction_level'] > 0.7).astype(int)

# Define features for Model 1
features_model_1 = ['last_evaluation', 'number_project', 'average_monthly_hours',
                     'left' ]
X1 = data[features_model_1]
y = data['satisfaction_level']

# Define features for Model 2
features_model_2 = ['last_evaluation', 'number_project', 'average_monthly_hours', 'left',
                    'average_hours_per_project', 'high_satisfaction']
X2 = data[features_model_2]

# Function to preprocess data and train model
def train_model(X, y, model_filename):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"Numerical columns: {numerical_cols}")

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])



    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])

    # Define the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_features': ['auto', 'sqrt', 'log2'],
        'regressor__max_depth': [None, 10, 20, 30]
    }

    # Perform GridSearch to find the best parameters
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the best model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE by setting squared=False
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Calculate MAPE

    # Display the evaluation metrics
    evaluation_results = {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Adjusted R-squared': adjusted_r2,
        'Mean Absolute Percentage Error': mape
    }

    print(f"Evaluation Results for {model_filename}:", evaluation_results)

    # Save the best model
    joblib.dump(best_model, model_filename)
    print(f"Model saved to {model_filename}")

    # Feature importance
    importances = best_model.named_steps['regressor'].feature_importances_

    # Ensure the feature names are correct
    preprocessor = best_model.named_steps['preprocessor']
    onehot_feature_names = []
    if 'cat' in dict(preprocessor.named_transformers_).keys():
        onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()

    feature_names = numerical_cols + onehot_feature_names
    print(f"Feature names: {feature_names}")
    print(f"Importances shape: {importances.shape}")
    print(f"Feature names shape: {len(feature_names)}")

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_names, importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance in Predicting Employee Satisfaction ({model_filename})')
    plt.show()

# Train and save Model 1
train_model(X1, y, 'model_1.pkl')

# Train and save Model 2
train_model(X2, y, 'model_2.pkl')


# In[45]:


from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the model
model_2 = load('model_2.pkl')

# Define a function to predict employee satisfaction
def predict_satisfaction(model, data):
    # Assuming data is a DataFrame containing the features
    prediction = model.predict(data)
    return prediction

# Define a function to calculate evaluation metrics
def calculate_metrics(actual, predicted, n_features):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    n = len(actual)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mse, rmse, mae, r2, adjusted_r2, mape

# Example usage
if __name__ == "__main__":
    # Load new data to make predictions
    new_data = pd.read_csv('HR_comma_sep.csv')

    # Assuming 'satisfaction_level' is the actual satisfaction score in the dataset
    if 'satisfaction_level' in new_data.columns:
        actual_satisfaction = new_data['satisfaction_level']
    else:
        actual_satisfaction = None

    # Drop the actual satisfaction level from the features if it exists
    if 'satisfaction_level' in new_data.columns:
        new_data = new_data.drop(columns=['satisfaction_level'])

    # Predict satisfaction levels using the model
    satisfaction_predictions_2 = predict_satisfaction(model_2, new_data)

    # Print the predictions
    print("Predictions from Model 2:", satisfaction_predictions_2)

    # Calculate and display evaluation metrics
    if actual_satisfaction is not None:
        n_features = new_data.shape[1]
        
        metrics_2 = calculate_metrics(actual_satisfaction, satisfaction_predictions_2, n_features)

        evaluation_results_2 = {
            'Mean Squared Error': metrics_2[0],
            'Root Mean Squared Error': metrics_2[1],
            'Mean Absolute Error': metrics_2[2],
            'R-squared': metrics_2[3],
            'Adjusted R-squared': metrics_2[4],
            'Mean Absolute Percentage Error': metrics_2[5]
        }

        print("\nEvaluation Metrics for Model 2:")
        for key, value in evaluation_results_2.items():
            print(f"{key}: {value}")

    # Visualization
    plt.figure(figsize=(12, 6))

    if actual_satisfaction is not None:
        # Plot the actual vs predicted satisfaction levels for Model 2
        plt.hist(actual_satisfaction, bins=20, color='lightgreen', edgecolor='black', alpha=0.7, label='Actual Satisfaction')
        plt.hist(satisfaction_predictions_2, bins=20, color='orange', edgecolor='black', alpha=0.5, label='Predicted Satisfaction (Model 2)')
        plt.title('Actual vs Predicted Satisfaction Levels (Model 2)')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

