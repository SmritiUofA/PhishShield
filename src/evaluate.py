import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/phishing_model.pkl")

# Load test data
df_test = pd.read_csv("/Users/hari/PhishShield/data/PhiUSIIL_Phishing_URL_Dataset.csv").sample(5)  # Take sample for testing
X_test = df_test.drop(columns=["label"])

# Make predictions
predictions = model.predict(X_test)

print("Predictions:", predictions)
