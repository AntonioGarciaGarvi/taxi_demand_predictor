# Ride Demand Predictor

End-to-end ML pipeline for predicting NYC taxi demand using MLOps best practices developed as part of Real-World Machine Learning course by Pau Labarta Bajo.

## Overview

This project builds a production-ready machine learning service to predict hourly taxi demand across NYC locations, helping ride-sharing companies optimize driver distribution and reduce lost revenue.

## Architecture

The system consists of three modular pipelines orchestrated via GitHub Actions and local Makefile commands:

### 🔄 Feature Pipeline
Transforms raw taxi data into time-series features and stores them in Hopsworks Feature Store. 

### 🧠 Training Pipeline  
Trains LightGBM models using historical features with Optuna hyperparameter optimization. 

### ⚡ Inference Pipeline
Generates hourly predictions using the latest production model.  

<p align="center">
<img src="https://antoniogarciagarvi.github.io/images/portfolio/taxi_demand_prediction/MLsystem.png" align="center">
</p>

## Tech Stack

- **ML**: LightGBM, Optuna, Scikit-learn
- **MLOps**: Hopsworks (Feature Store), Comet ML (Model Registry)
- **Visualization**: Streamlit, Plotly, PyDeck
- **Infrastructure**: GitHub Actions, Poetry
- **Data**: Pandas, NumPy

## Quick Start

1. **Setup environment**
   ```bash
   make init
   ```

2. **Configure credentials**
   Create `.env` file with Hopsworks and Comet ML keys.

3. **Run pipelines**
   ```bash
   make backfill     # Populate historical data (one-time setup)
   make features     # Generate features
   make training     # Train model  
   make inference    # Generate predictions for the last hour
   ```

4. **Launch dashboards**
   ```bash
   make frontend-app     # Prediction dashboard 
   make monitoring-app   # Model monitoring 
   ```

## Key Features

- **Feature Store**: Decouples feature engineering from model training with 672-hour sliding window features 
- **Model Registry**: Automatic model promotion based on MAE threshold (< 30.0) 
- **Monitoring**: Real-time MAE tracking with drift detection
- **Automation**: GitHub Actions workflows for production deployment

## Project Structure

```
├── data/                  # Data storage  
│   ├── raw/              # Raw NYC taxi data  
│   ├── transformed/      # Processed time-series data  
│   └── cache/            # Cached intermediate files  
├── models/               # Trained model artifacts  
├── scripts/              # Pipeline implementations  
│   ├── feature_pipeline.py  
│   ├── training_pipeline.py  
│   ├── inference_pipeline.py  
│   ├── backfill_feature_group.py  
│   └── backfill_inference.py  
├── src/                  # Core modules and APIs  
│   ├── config.py  
│   ├── data.py  
│   ├── feature_store_api.py  
│   ├── model_registry_api.py  
│   ├── inference.py  
│   ├── frontend.py  
│   └── frontend_monitoring.py  
├── notebooks/            # Development notebooks     
├── .github/workflows/    # CI/CD automation  
│   ├── feature_pipeline.yaml  
│   ├── training_pipeline.yaml  
│   └── inference_pipeline.yaml  
├── Makefile              # Local development commands  
└── pyproject.toml        # Poetry dependencies  
```




