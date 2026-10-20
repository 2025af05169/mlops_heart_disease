# Heart Disease Prediction API - MLOps Pipeline

This repository implements an end-to-end production-ready Machine Learning system for predicting heart disease risk based on clinical parameters.

## Project Overview & Architecture
* **Machine Learning Model**: Random Forest Classifier (**AUC: 95.18%**)
* **API Framework**: FastAPI with Swagger UI
* **Testing Suite**: Pytest (Automated Unit & Integration Tests)
* **Containerization**: Docker
* **CI/CD Pipeline**: GitHub Actions (Automated Linting, Training & Pytest)
* **Monitoring**: Prometheus Metrics Instrumentation

## Repository Structure
* `src/`: Data loading, model training, and FastAPI deployment code.
* `tests/`: Automated unit test assertions.
* `models/`: Saved model artifacts.
* `data/`: Raw and processed dataset files.
* `.github/workflows/`: CI/CD automation definition (`main.yml`).
* `screenshot/`: Execution proof and pipeline verification logs.

## Quickstart Guide

### 1. Run Locally
```bash
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.app:app --reload
At chrome open : http://127.0.0.1:8000/
