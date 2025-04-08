import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = 'hospital_daily.csv'
data = pd.read_csv(file_path)

# Preprocessing: Convert Date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# List of wards
wards = [
    "Accident and Emergency", "ICU", "Pediatric", "Maternity",
    "General", "Acute Care", "Orthopedic", "Executive and VVIP"
]

# Dictionary to store models and accuracies
ward_models = {}
ward_accuracies = {}

# Train and evaluate a model for each ward
for ward in wards:
    # Define features (X) and target (y) for the current ward
    X = data.drop(columns=['Date', ward])
    y = data[ward]
    
    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Convert R² score to percentage
    
    # Store the model and accuracy
    ward_models[ward] = model
    ward_accuracies[ward] = accuracy

# Display accuracies for each ward
print("Accuracy for each ward (%):")
for ward, accuracy in ward_accuracies.items():
    print(f"{ward}: {accuracy:.2f}%")

# Take user input for date and predict number of beds for each ward
user_date = input("\nEnter a date (YYYY-MM-DD): ")
user_date = pd.to_datetime(user_date)

# Find the closest matching row in the dataset based on the user's date
if user_date in data['Date'].values:
    input_features = data.loc[data['Date'] == user_date].drop(columns=['Date', 'Admission'])
else:
    print("Date not found in the dataset. Please enter a valid date.")
    exit()

# Predict the number of beds for each ward
print("\nPredicted number of beds for each ward:")
predictions = {}
for ward, model in ward_models.items():
    input_data = input_features.drop(columns=ward)  # Exclude the current ward from features
    prediction = model.predict(input_data)[0]  # Predict for the provided date
    predictions[ward] = prediction
    print(f"{ward}: {prediction:.2f}")

# Optionally, display all predictions in a tabular format
predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Predicted Beds"])
print("\nPredicted Beds per Ward:")
print(predictions_df)
