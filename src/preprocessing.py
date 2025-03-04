import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """Preprocess dataset (handle missing values, scaling, etc.)."""
    df = pd.read_csv("/Users/hari/PhishShield/data/PhiUSIIL_Phishing_URL_Dataset.csv")
    # Drop FILENAME and URL as they are not features
    df.drop(columns=["FILENAME", "URL"], inplace=True)

    # Encode categorical columns
    categorical_cols = ["TLD", "Domain", "Title"]
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

    # Fill missing values
    df.fillna(0, inplace=True)

    # Normalize numeric features
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

if __name__ == "__main__":
    df = preprocess_data("/Users/hari/PhishShield/data/PhiUSIIL_Phishing_URL_Dataset.csv")
