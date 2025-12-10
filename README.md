Stock Prediction LSTM 📈
======================

## Project Overview

This repository contains the solution for the Tech Challenge - Phase 4 of the Machine Learning Engineering post-graduate program. The goal is to build a Deep Learning predictive model using an LSTM (Long Short-Term Memory) architecture to forecast stock closing prices, delivering the results through a containerized RESTful API.

## 🗂️ Project Structure

The project follows a modular monolith architecture to streamline collaboration and artifact sharing between the modeling and engineering stages.

```
/
├── notebooks/          # Exploratory analysis (EDA), experiments, and visuals
├── src/
│   ├── model_training/ # Python scripts to train and persist the model
│   └── api/            # API code (main.py, schemas, routes)
├── artifacts/          # Storage for the trained model and scaler
│   ├── model.pt        # Serialized model
│   └── scaler.pkl      # Scaler object for normalization/denormalization
├── Dockerfile          # Docker image definition for the API
├── requirements.txt    # Project dependencies
└── README.md           # Main documentation
```

## 🛠️ Technology Stack

- Python for scripts and API implementation
- yfinance for data ingestion
- PyTorch (LSTM) for modeling
- Flask for serving the API
- Docker for containerization
