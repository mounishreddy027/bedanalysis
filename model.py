import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = 'hospital_daily.csv'
data = pd.read_csv(file_path)

# Preprocess the dataset
data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month

# Select features and target variables (all wards)
features = ['Year', 'Month']
wards = [
    'Accident and Emergency', 'ICU', 'Pediatric', 'Maternity',
    'General', 'Acute Care', 'Orthopedic', 'Executive and VVIP'
]

# Split data into training and testing sets
X = data[features]
y = data[wards]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model for each ward
models = {}
rmse_scores = {}
accuracy_scores = {}

for ward in wards:
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train[ward])
    models[ward] = model

    # Predict on the test set
    y_pred = model.predict(X_test)
    rmse_scores[ward] = np.sqrt(mean_squared_error(y_test[ward], y_pred))
    
    # Calculate accuracy percentage
    accuracy = 100 - (np.mean(np.abs((y_test[ward] - y_pred) / y_test[ward])) * 100)
    accuracy_scores[ward] = accuracy

# Display RMSE and accuracy scores
print("Performance Metrics for Each Ward:")
for ward in wards:
    print(f"{ward}:")
    print(f"  RMSE: {rmse_scores[ward]:.2f}")
    print(f"  Accuracy: {accuracy_scores[ward]:.2f}%")

# Function to predict the number of beds for a given date
def predict_beds(input_date):
    # Convert input date to a pandas datetime object
    input_date = pd.to_datetime(input_date)
    input_year = input_date.year
    input_month = input_date.month

    # Create a DataFrame for the input features
    input_features = pd.DataFrame({'Year': [input_year], 'Month': [input_month]})

    # Predict the number of beds for each ward
    predictions = {}
    for ward, model in models.items():
        predictions[ward] = model.predict(input_features)[0]

    return predictions

# Example: Predict beds for a given date
input_date = input("\nEnter a date (YYYY-MM-DD): ")
predicted_beds = predict_beds(input_date)

# Display predictions
print("\nPredicted Number of Beds for Each Ward:")
for ward, beds in predicted_beds.items():
    print(f"{ward}: {beds:.2f}")
