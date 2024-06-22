from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = load('best_model.joblib')

# Define a function to predict employee satisfaction
def predict_satisfaction(data):
    # Assuming data is a DataFrame containing the features
    prediction = model.predict(data)
    return prediction

# Example usage
if __name__ == "__main__":
    # Load new data to make predictions
    new_data = pd.read_csv('path_to_new_data.csv')
    satisfaction_predictions = predict_satisfaction(new_data)
    print(satisfaction_predictions)
