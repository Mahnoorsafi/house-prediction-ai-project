import kagglehub
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

# Construct file path (assuming CSV format)
csv_file_path = f"{dataset_path}/Housing.csv"

# Load dataset
data = pd.read_csv(csv_file_path)

# Check column names to avoid KeyError
print("Dataset Columns:", data.columns)

# Selecting relevant features for prediction
features = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
    'guestroom', 'basement', 'hotwaterheating', 'airconditioning'
]

X = data[features]  # Independent variables
y = data['price']   # Dependent variable (house price)

# Convert categorical variables to numerical (0 or 1)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Display sample predictions
sample_data = X_test.iloc[:5]
sample_predictions = model.predict(sample_data)
print("\nSample Predictions:")
for i, pred in enumerate(sample_predictions):
    print(f"House {i+1}: Predicted Price = {pred:.2f}")

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'model.pkl'")