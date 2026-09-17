# 🚀 Order Delivery Inference Service (MLOps Task 3)

## 📌 Overview
This project transforms experimental Jupyter notebooks into a production-grade inference service. It provides a RESTful API to predict order delivery status (`Late` vs. `On time`) along with prediction probabilities, using saved fitted transformers and registered models without re-training at inference time.

## 🛠 Tech Stack & Tools
* **Data Versioning:** DVC
* **Data Validation:** Great Expectations
* **Experiment Tracking & Model Registry:** MLflow
* **API Framework:** FastAPI
* **Containerization:** Docker & Docker Compose
* **Testing & Quality Assurance:** Pytest & Pre-commit Hooks
* **CI/CD:** GitHub Actions

## 🏗 Key Features & Architecture
- **Notebooks to Modules:** Refactored pipelines into clean Python modules (`src/`) covering data access, validation, preprocessing, and prediction.
- **Data & Model Lineage:** Dataset versioning via `DVC` and model tracking/versioning using `MLflow Model Registry`.
- **Data Validation:** Schema and range validation on incoming order data using `Great Expectations` to handle missing/malformed payloads safely.
- **FastAPI Service:**
  - `GET /health` - Health check route.
  - `GET /info` - Model info and version details.
  - `POST /predict` - Single & batch order predictions (returns prediction, probability, and model version).
- **Automated Testing & CI/CD:** Unit, data, and integration tests with `pytest` triggered automatically via `GitHub Actions` on every push.

## 🗂 Project Structure
```text
├── app/                  # FastAPI app and API schemas
├── config/               # Configuration files (no hardcoded paths)
├── data/                 # DVC tracked data
├── models/               # Saved fitted artifacts & model registry links
├── notebooks/            # Original exploratory notebooks
├── src/                  # Core Python modules (preprocessing, features, prediction)
├── tests/                # Unit & integration tests
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
└── README.md
