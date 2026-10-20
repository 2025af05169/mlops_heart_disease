import os
import pickle

def test_model_artifact_exists():
    assert os.path.exists("models/best_model.pkl")
    assert os.path.exists("models/scaler.pkl")

def test_model_prediction():
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Test sample patient with 13 health features
    sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    pred = model.predict(sample_data)
    assert pred[0] in [0, 1]