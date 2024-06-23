#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Load the dataset
file_path = 'HR_comma_sep.csv'
data = pd.read_csv(file_path)

# Correct the column names
data.rename(columns={
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'time_spent_company',
    'Department': 'department'
}, inplace=True)

# Define features and target
X = data.drop(columns=['satisfaction_level'])
y = data['satisfaction_level']

# Define categorical and numerical columns
categorical_cols = ['department', 'salary']
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

# Feature Engineering: Creating new feature
X['average_monthly_hours_per_year'] = X['average_monthly_hours'] * 12

# Update numerical columns to include the new feature
numerical_cols.append('average_monthly_hours_per_year')


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Determine the number of rows needed based on the number of numerical columns
num_cols = 4  # Number of figures per row
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols  # Calculate the number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

# Flatten axes array if there are multiple rows
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(X[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Remove any empty subplots if there are fewer columns than grid slots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[5]:


# Visualizing categorical features
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=X)
    plt.title(f'Count of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[6]:


# Visualizing correlation matrix for numerical features
plt.figure(figsize=(12, 8))
corr_matrix = X[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features')
plt.show()


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
                    'left', ]
X1 = data[features_model_1]
y = data['satisfaction_level']

# Define features for Model 2
features_model_2 = ['last_evaluation', 'number_project', 'average_monthly_hours',
                     'left', 'average_hours_per_project', 'high_satisfaction']
X2 = data[features_model_2]

# Function to preprocess data and train model
def train_model(X, y, model_filename):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Numerical columns: {numerical_cols}")

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Bundle preprocessing for numerical data
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
    feature_names = numerical_cols
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


# In[8]:


from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the models
model_1 = load('model_1.pkl')
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
    new_data = pd.read_csv("HR_comma_sep.csv")

    # Assuming 'satisfaction_level' is the actual satisfaction score in the dataset
    if 'satisfaction_level' in new_data.columns:
        actual_satisfaction = new_data['satisfaction_level']
    else:
        actual_satisfaction = None

    # Drop the actual satisfaction level from the features if it exists
    if 'satisfaction_level' in new_data.columns:
        new_data = new_data.drop(columns=['satisfaction_level'])

    # Predict satisfaction levels using both models
    satisfaction_predictions_1 = predict_satisfaction(model_1, new_data)
    satisfaction_predictions_2 = predict_satisfaction(model_2, new_data)

    # Print the predictions
    print("Predictions from Model 1:", satisfaction_predictions_1)
    print("Predictions from Model 2:", satisfaction_predictions_2)

    # Calculate and display evaluation metrics
    if actual_satisfaction is not None:
        n_features = new_data.shape[1]

        metrics_1 = calculate_metrics(actual_satisfaction, satisfaction_predictions_1, n_features)
        metrics_2 = calculate_metrics(actual_satisfaction, satisfaction_predictions_2, n_features)

        evaluation_results_1 = {
            'Mean Squared Error': metrics_1[0],
            'Root Mean Squared Error': metrics_1[1],
            'Mean Absolute Error': metrics_1[2],
            'R-squared': metrics_1[3],
            'Adjusted R-squared': metrics_1[4],
            'Mean Absolute Percentage Error': metrics_1[5]
        }

        evaluation_results_2 = {
            'Mean Squared Error': metrics_2[0],
            'Root Mean Squared Error': metrics_2[1],
            'Mean Absolute Error': metrics_2[2],
            'R-squared': metrics_2[3],
            'Adjusted R-squared': metrics_2[4],
            'Mean Absolute Percentage Error': metrics_2[5]
        }

        print("\nEvaluation Metrics for Model 1:")
        for key, value in evaluation_results_1.items():
            print(f"{key}: {value}")

        print("\nEvaluation Metrics for Model 2:")
        for key, value in evaluation_results_2.items():
            print(f"{key}: {value}")

    # Visualization
    plt.figure(figsize=(12, 12))

    if actual_satisfaction is not None:
        # Plot the actual vs predicted satisfaction levels for Model 1
        plt.subplot(2, 1, 1)
        plt.hist(actual_satisfaction, bins=20, color='lightgreen', edgecolor='black', alpha=0.7, label='Actual Satisfaction')
        plt.hist(satisfaction_predictions_1, bins=20, color='skyblue', edgecolor='black', alpha=0.5, label='Predicted Satisfaction (Model 1)')
        plt.title('Actual vs Predicted Satisfaction Levels (Model 1)')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')



        # Plot the actual vs predicted satisfaction levels for Model 2
        plt.subplot(2, 1, 2)
        plt.hist(actual_satisfaction, bins=20, color='lightgreen', edgecolor='black', alpha=0.7, label='Actual Satisfaction')
        plt.hist(satisfaction_predictions_2, bins=20, color='orange', edgecolor='black', alpha=0.5, label='Predicted Satisfaction (Model 2)')
        plt.title('Actual vs Predicted Satisfaction Levels (Model 2)')
        plt.xlabel('Satisfaction Level')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# In[10]:


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

