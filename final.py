import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from ta import trend, momentum, volatility
import pickle
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_ as xavier_uniform
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from itertools import product
import json  # For JSON serialization
import sys   # To read input from stdin
import runpod
import base64  # Added for encoding predictions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Metrics Dictionary
metrics_dict = {
    "status": "pending",
    "message": "",
    "details": {}
}

# Configuration - Securely load API keys using environment variables
API_KEY = 'de_kgSuhw6v4KnRK0wprJCoBAIhqSd5R'  # Replace with your actual API key
BASE_URL = 'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}'

# List of cryptocurrencies (tickers) you want to collect data for
TICKERS = [
    'X:AAVEUSD',
    'X:AVAXUSD',
    'X:BATUSD',
    'X:LINKUSD',
    'X:UNIUSD',
    'X:SUSHIUSD',
    'X:PNGUSD',
    'X:JOEUSD',
    'X:XAVAUSD',
    'X:ATOMUSD',
    'X:ALGOUSD',
    'X:ARBUSD',
    'X:1INCHUSD',
    'X:DAIUSD',
    # Add more tickers as needed
]

# Timeframes you want to collect data for
TIMEFRAMES = [
    {'multiplier': 1, 'timespan': 'second'},
]

DATA_DIR = 'crypto_data'
os.makedirs(DATA_DIR, exist_ok=True)

MODELS_DIR = 'models'
SCALERS_DIR = 'scalers'
PREDICTIONS_DIR = 'predictions'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ... [All other classes and functions remain unchanged] ...

def preprocess_and_predict(crypto_metrics):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data_dir = 'preprocessed_data'
    cryptos = [d for d in os.listdir(preprocessed_data_dir) if os.path.isdir(os.path.join(preprocessed_data_dir, d))]

    all_predictions = []
    for crypto in cryptos:
        ticker_dir = os.path.join(preprocessed_data_dir, crypto)
        csv_files = [f for f in os.listdir(ticker_dir) if f.endswith('_preprocessed.csv')]
        if not csv_files:
            logging.warning(f"No preprocessed data found for prediction: {crypto}. Skipping.")
            continue

        df_list = []
        for f in csv_files:
            fp = os.path.join(ticker_dir, f)
            try:
                df = pd.read_csv(fp)
                df['crypto'] = crypto
                df_list.append(df)
            except Exception as e:
                logging.error(f"Error reading {fp}: {e}")
                continue
        if not df_list:
            continue

        full_df = pd.concat(df_list, ignore_index=True)
        full_df.dropna(subset=['close_price'], inplace=True)
        if len(full_df) <= 80:
            logging.warning(f"Not enough data for prediction: {crypto}. Skipping.")
            continue

        # Load scaler for this crypto
        scaler_path = os.path.join(SCALERS_DIR, f'{crypto}_scaler.joblib')
        if not os.path.exists(scaler_path):
            logging.warning(f"No scaler found for {crypto}. Skipping prediction.")
            continue
        scaler = joblib.load(scaler_path)
        full_df['close_price'] = scaler.transform(full_df[['close_price']])

        # Dataset and DataLoader
        test_dataset = CryptoDataset(full_df, window_size=80)
        if len(test_dataset) == 0:
            logging.warning(f"No samples for prediction: {crypto}. Skipping.")
            continue
        test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=2)

        # Load model
        model_path = os.path.join(MODELS_DIR, f'best_liteformer_model_{crypto}.pth')
        if not os.path.exists(model_path):
            logging.warning(f"No model found for {crypto}. Skipping prediction.")
            continue

        model = LiteFormer(
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=80
        ).to(device)

        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            logging.error(f"Error loading model for {crypto}: {e}")
            continue

        predictions = []
        prediction_timestamps = []

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                outputs = outputs.cpu().numpy().flatten()
                predictions.extend(outputs.tolist())

                start_index = len(predictions) - len(outputs)
                end_index = start_index + len(outputs)
                # Align timestamps
                corresponding_timestamps = full_df['t'].iloc[start_index + 80:end_index + 80].tolist()
                prediction_timestamps.extend(corresponding_timestamps)

        # Create predictions_df
        predictions_df = pd.DataFrame({
            't': prediction_timestamps,
            'crypto': full_df['crypto'].iloc[80:80 + len(predictions)].tolist(),
            'predicted_close_price': predictions
        })

        # Inverse transform predictions
        predictions_df['predicted_close_price'] = scaler.inverse_transform(predictions_df[['predicted_close_price']])

        # Retrieve true close prices for comparison
        true_close_prices = full_df['close_price'].iloc[80:80 + len(predictions)]
        true_close_prices = scaler.inverse_transform(true_close_prices.values.reshape(-1, 1)).flatten()

        # Compute metrics
        if len(true_close_prices) > 0:
            mae = mean_absolute_error(true_close_prices, predictions_df['predicted_close_price'])
            rmse = math.sqrt(mean_squared_error(true_close_prices, predictions_df['predicted_close_price']))
        else:
            mae = 0
            rmse = 0

        print(f"Prediction MAE for {crypto}: {mae:.4f}, RMSE: {rmse:.4f}")

        # Store prediction metrics
        if crypto in crypto_metrics:
            crypto_metrics[crypto]["prediction_metrics"] = {
                "mae": mae,
                "rmse": rmse
            }
        else:
            crypto_metrics[crypto] = {
                "prediction_metrics": {
                    "mae": mae,
                    "rmse": rmse
                }
            }

        # Add classification columns
        last_actual_close = true_close_prices
        predictions_df['direction'] = np.where(predictions_df['predicted_close_price'] > last_actual_close, 'Up', 'Down')
        predictions_df['percentage_change'] = ((predictions_df['predicted_close_price'] - last_actual_close) / last_actual_close) * 100
        predictions_df['last_actual_close'] = last_actual_close

        def categorize_change(pc):
            if pc >= 2:
                return 'Significant Up'
            elif 0.5 <= pc < 2:
                return 'Moderate Up'
            elif -0.5 < pc < 0.5:
                return 'No Change'
            elif -2 < pc <= -0.5:
                return 'Moderate Down'
            else:
                return 'Significant Down'

        predictions_df['change_category'] = predictions_df['percentage_change'].apply(categorize_change)
        predictions_df['percentage_change'] = predictions_df['percentage_change'].round(2)

        # Save per-crypto predictions
        crypto_prediction_path = os.path.join(PREDICTIONS_DIR, f'{crypto}_latest_predictions.csv')
        predictions_df.to_csv(crypto_prediction_path, index=False)
        logging.info(f"Predictions for {crypto} saved to {crypto_prediction_path}")

        all_predictions.append(predictions_df)

    # Merge all predictions into one file if desired
    if all_predictions:
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)
        final_path = os.path.join(PREDICTIONS_DIR, 'all_latest_predictions.csv')
        final_predictions_df.to_csv(final_path, index=False)
        logging.info(f"All predictions combined saved to {final_path}")
        print(final_predictions_df.head())

    return {"status": "success", "message": "Predictions completed.", "predictions_path": final_path}

def handler(job):
    job_input = job.get("input", {})
    
    # Retrieve the 'START_DATE' and 'END_DATE' values
    START_DATE1 = job_input.get("START_DATE1", "")
    print(f"START_DATE1 :  {START_DATE1}")
    END_DATE1 = job_input.get("END_DATE1", "")
    print(f"END_DATE1 :  {END_DATE1}")

    collect(START_DATE1, END_DATE1)
    # Uncomment if you want to run these steps
    preprocess = CryptoDataPreprocessor(
         raw_data_dir='crypto_data',
         preprocessed_data_dir='preprocessed_data',
         columns_to_add=['close_price', 't']
     )
    preprocess.preprocess_all_files()

    crypto_metrics = {}
    main(crypto_metrics)

    START_DATE2 = job_input.get("START_DATE2", "")
    print(f"START_DATE2 :  {START_DATE2}")
    END_DATE2 = job_input.get("END_DATE2", "")
    print(f"END_DATE2 :  {END_DATE2}")

    collect(START_DATE2, END_DATE2)
    preprocess = CryptoDataPreprocessor(
        raw_data_dir='crypto_data',
        preprocessed_data_dir='preprocessed_data',
        columns_to_add=['close_price', 't']
    )
    preprocess.preprocess_all_files()

    # Step 2: Predictions
    prediction_status = preprocess_and_predict(crypto_metrics)  # Pass crypto_metrics to collect prediction metrics
    if prediction_status["status"] != "success":
        logging.error("Prediction step failed. Exiting.")
        metrics_dict["status"] = "failed"
        metrics_dict["message"] = "Prediction step failed."
        print(json.dumps(metrics_dict))
        return json.dumps(metrics_dict)  # Ensure handler exits after failure

    # Step 3: Load the combined predictions
    all_predictions_path = os.path.join(PREDICTIONS_DIR, 'all_latest_predictions.csv')
    if not os.path.exists(all_predictions_path):
        logging.error(f"Combined predictions file not found at {all_predictions_path}. Exiting.")
        metrics_dict["status"] = "failed"
        metrics_dict["message"] = f"Combined predictions file not found at {all_predictions_path}."
        print(json.dumps(metrics_dict))
        return json.dumps(metrics_dict)  # Ensure handler exits after failure

    predictions_df = pd.read_csv(all_predictions_path)

    # Step 4: Encode predictions_df as Base64
    predictions_csv = predictions_df.to_csv(index=False)
    encoded_predictions = base64.b64encode(predictions_csv.encode('utf-8')).decode('utf-8')

    # Step 5: Aggregate Metrics
    metrics_dict["status"] = "success"
    metrics_dict["message"] = "Processing completed successfully."
    metrics_dict["details"] = crypto_metrics
    metrics_dict["predictions_csv"] = encoded_predictions  # Add encoded predictions to response

    return json.dumps(metrics_dict)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# =========================
# Main Execution
# =========================

runpod.serverless.start({"handler": handler})
