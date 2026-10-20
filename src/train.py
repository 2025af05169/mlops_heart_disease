import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def train_and_evaluate():
    mlflow.set_experiment("Heart_Disease_Prediction")

    df = pd.read_csv("data/processed/heart_clean.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_auc = 0
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "LogisticRegression":
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", auc)

            print(f"[{name}] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_model_name = name

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best Model Saved: {best_model_name} with AUC = {best_auc:.4f}")

if __name__ == "__main__":
    train_and_evaluate()