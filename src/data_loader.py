import os
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/dataprofessor/data/master/heart-disease-cleveland.csv"
COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

def load_and_clean_data(raw_path="data/raw/heart.csv", processed_path="data/processed/heart_clean.csv"):
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)

    if not os.path.exists(raw_path):
        df = pd.read_csv(DATA_URL)
        if len(df.columns) == len(COLUMNS):
            df.columns = COLUMNS
        df.to_csv(raw_path, index=False)
    else:
        df = pd.read_csv(raw_path)

    # Impute missing values with median
    for col in ['ca', 'thal']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # Convert multi-class target to binary (0 = No Disease, 1 = Disease)
    if 'target' in df.columns:
        df['target'] = (df['target'] > 0).astype(int)

    df.to_csv(processed_path, index=False)
    print("Data loaded and preprocessed successfully.")
    return df

if __name__ == "__main__":
    load_and_clean_data()