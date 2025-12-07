# Ride Demand Predictor

End-to-end ML pipeline for predicting NYC taxi demand using MLOps best practices.

## Overview

This project builds a production-ready machine learning service to predict hourly taxi demand across NYC locations, helping ride-sharing companies optimize driver distribution and reduce lost revenue. [1](#0-0) 

## Architecture

The system consists of three modular pipelines orchestrated via GitHub Actions and local Makefile commands:

### 🔄 Feature Pipeline
Transforms raw taxi data into time-series features and stores them in Hopsworks Feature Store. 

### 🧠 Training Pipeline  
Trains LightGBM models using historical features with Optuna hyperparameter optimization. 

### ⚡ Inference Pipeline
Generates hourly predictions using the latest production model.  

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
   make features     # Generate features
   make training     # Train model  
   make inference    # Generate predictions
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
├── scripts/           # Pipeline implementations
├── src/              # Core modules and APIs
├── notebooks/        # Development notebooks
├── .github/workflows/ # CI/CD automation
└── Makefile          # Local development commands
```




