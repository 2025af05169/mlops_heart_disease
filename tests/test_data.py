import os
import pandas as pd
from src.data_loader import load_and_clean_data

def test_data_processing():
    df = load_and_clean_data()
    assert os.path.exists("data/processed/heart_clean.csv")
    assert df.isnull().sum().sum() == 0
    assert set(df['target'].unique()).issubset({0, 1})