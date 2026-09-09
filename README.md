# MLOps Task 3 Submission 🚀

This repository contains the advanced implementation for **Task 3** (built on top of previous pipeline workflows), focusing on an end-to-end MLOps architecture, model tracking, data versioning with DVC, and automated CI/CD testing using GitHub Actions.

## 📌 Project Overview
* **Data Processing & Pipeline:** Automated data ingestion, feature engineering, and preprocessing workflows using Python, Pandas, and Scikit-Learn.
* **Model Tracking:** Experiment tracking, metrics logging, and artifact management.
* **Data Versioning:** Managed via DVC (`.dvc`) to track dataset versions seamlessly across iterations.
* **Automated Testing:** Unit tests integrated using `pytest` to ensure code reliability and robustness in CI pipelines.

## 🛠️ Project Structure
```text
├── .github/workflows/ci.yml   # GitHub Actions CI/CD automated pipeline
├── .dvc/                      # Data Version Control configuration
├── config/                    # Configuration files (config.yaml)
├── mlruns/                    # Experiment tracking artifacts
├── outputs/ & logs/           # Model predictions and execution logs
├── src/                       # Source code modules
├── test_pipeline.py           # Unit tests for pytest execution
└── README.md                  # Project documentation
