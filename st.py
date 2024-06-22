from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os




# Verify the file path
file_path = 'HR_comma_sep.csv'
if not os.path.isfile(HR):
    raise FileNotFoundError(f"The file {HR_comma_sep.csv} does not exist.")



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
