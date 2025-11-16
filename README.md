Madrid House Price Prediction — MLOps Project

This project develops an end-to-end MLOps pipeline for predicting house prices in Madrid. It includes data preprocessing, model training and evaluation, API serving with FastAPI, containerization using Docker, and automated deployment through GitHub Actions and Render.

Project Overview
1. Data and Training

The pipeline processes a housing dataset and trains a RandomForestRegressor model. Model metrics and parameters are logged using MLflow for experiment tracking.

2. Model Serving

A FastAPI application provides an HTTP interface for real-time price predictions based on property features.

3. Containerization

The serving application and trained model are packaged into a Docker image for portability and consistency.

4. CI/CD Automation

GitHub Actions workflows automate model training, artifact management, Docker image builds, and deployment.

5. Deployment

The project is deployed on Render, which automatically builds and runs the Docker container when updates are pushed to the main branch.

Repository Structure
madrid_mlops_project/
├── app/
│   └── app.py                  # FastAPI app for serving predictions
├── data/
│   └── madrid_sample.csv       # Sample dataset for CI testing
├── models/
│   └── model.pkl               # Trained model artifact
├── src/
│   ├── data_cleaning.py        # Data preprocessing script
│   └── train_model.py          # Model training and MLflow logging
├── .github/workflows/
│   ├── train.yml               # Training workflow
│   └── docker.yml              # Docker build and deploy workflow
├── Dockerfile                  # Container build instructions
├── render.yaml                 # Render deployment configuration
├── requirements-train-ci.txt   # Training dependencies (for CI)
├── requirements-api.txt        # Runtime dependencies (for FastAPI)
└── README.md                   # Project documentation

Local Setup
1. Create and activate a virtual environment
  python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux

2.Install dependencies
For training:
pip install -r requirements-train-ci.txt
For serving:
pip install -r requirements-api.txt

3. Train the model locally
python src/train_model.py data/house_price_madrid_14_08_2022.csv
The trained model will be saved to models/model.pkl. MLflow logs will be available in the mlruns/ directory.

4. Run the API locally
uvicorn app.app:app --host 0.0.0.0 --port 8000
Then open http://127.0.0.1:8000/docs to test the endpoints.

Example Prediction

Endpoint: POST /predict
Request body:
[
  {
    "m2": 85,
    "rooms": 3,
    "elevator": 1,
    "garage": 0,
    "house_type": "piso",
    "house_type_2": "reformado",
    "neighborhood": "sol",
    "district": "centro"
  }
]
Response:
{
  "count": 1,
  "log_price": [11.05],
  "price": [63000.42]
}

CI/CD Workflows
train.yml
- Runs on every push to the main branch
- Installs lightweight dependencies
- Trains on a sample dataset (data/madrid_sample.csv)
- Uploads models/model.pkl as a workflow artifact

docker.yml
- Downloads the model artifact
- Builds and tests the Docker image
- Pushes the image to GitHub Container Registry (GHCR)
- Optionally performs a smoke test

Deployment on Render
Configuration:
| Setting           | Value    |
| ----------------- | -------- |
| Runtime           | Docker   |
| Branch            | main     |
| Port              | 8000     |
| Health Check Path | /healthz |
| Auto Deploy       | Enabled  |

Endpoints:
- /healthz – service health check
-/docs – Swagger UI for API documentation
- /predict – prediction endpoint
Live deployment: https://madrid-house-price-api.onrender.com

Tools and Technologies
| Category            | Tools                        |
| ------------------- | ---------------------------- |
| Data & Modeling     | pandas, scikit-learn, joblib |
| Experiment Tracking | MLflow                       |
| API                 | FastAPI, Uvicorn             |
| CI/CD               | GitHub Actions, Docker, GHCR |
| Deployment          | Render.com                   |
