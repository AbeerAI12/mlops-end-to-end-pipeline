# 🚀 Order Delivery Inference Service

## 📌 Overview
This project transforms an end-to-end Machine Learning pipeline for order delivery prediction into a production-ready inference service. It covers data version control, experiment tracking, API deployment with FastAPI, containerization, and automated CI/CD workflows.

## 🛠 Tech Stack & Tools
* **Data Version Control (DVC):** Versioning datasets (`.parquet`) and pipeline models (`.pkl.dvc`).
* **Experiment Tracking:** MLflow (`mlflow.db` & `mlruns`).
* **API Framework:** FastAPI (`app.py`).
* **Containerization:** Docker & Docker Compose (`Dockerfile`, `docker-compose.yml`).
* **Testing:** Pytest (`test_pipeline.py`).
* **CI/CD:** GitHub Actions workflows (`.github/workflows`).

## 🏗 Key Features
- **Data & Model Versioning:** Tracked feature processing, datasets, and saved artifacts (`imputer`, `scaler`, `final_model`) using `DVC`.
- **Experiment Logging:** Recorded pipeline runs and metrics using `MLflow`.
- **RESTful API Service:** Built lightweight endpoint serving via `FastAPI` (`app.py`).
- **Containerization:** Containerized application setup with `Docker` and orchestrated using `docker-compose`.
- **Automated Testing & CI/CD:** Integrated `pytest` suite running automatically through `GitHub Actions` pipelines.

## 🗂 Project Structure
```text
├── .dvc/                   # DVC configuration
├── config/                 # Configuration files
├── mlruns/ & mlflow.db     # MLflow tracking data & database
├── outputs/                # Generated artifacts
├── *.ipynb                 # Pipeline development notebooks (01 to 06)
├── app.py                  # FastAPI application entrypoint
├── pipeline.py             # Inference & pipeline logic
├── test_pipeline.py        # Automated test suite (pytest)


## 🚀 How to Run

1. Clone the repo:
   `git clone https://github.com/AbeerAl12/mlops-end-to-end-pipeline.git`

2. Run with Docker:
   `docker-compose up --build`


├── Dockerfile              # Container configuration
├── docker-compose.yml      # Service orchestration
├── *.dvc                   # DVC tracking pointers for model artifacts & data
└── README.md               # Project documentation
