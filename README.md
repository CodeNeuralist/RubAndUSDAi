# USD to RUB Exchange Rate Prediction using LSTM

This project utilizes a Long Short-Term Memory (LSTM) neural network to predict the USD to RUB exchange rate based on historical data. The LSTM model is implemented using PyTorch, a popular deep learning framework. The dataset used for training and testing the model is fetched using the Yahoo Finance API (`yfinance`) and preprocessed using Min-Max scaling.

## Dependencies
- torch
- numpy
- yfinance
- scikit-learn

## Dataset
The dataset consists of historical USD to RUB exchange rate data obtained from Yahoo Finance (`USDRUB=X`) spanning from March 18, 2023, to March 18, 2024.

## Model Architecture
The LSTM model architecture is defined as follows:
- Input size: 1
- Hidden size: 64
- Output size: 1
- Number of LSTM layers: 2
- Sequence length: 7

## Training
The model is trained using mean squared error (MSE) loss and optimized using the Adam optimizer. Training is carried out for 100 epochs.

## Evaluation
After training, the model is evaluated by making predictions for the next day's exchange rate. The predicted value is then inverse transformed to obtain the actual exchange rate prediction.

## Files
- `main.py`: Python script containing the code for data preprocessing, model definition, training, and evaluation.
- `README.md`: This file, providing an overview of the project, dependencies, dataset, model architecture, training process, and evaluation.

## Usage
To run the code:
```bash
python main.py
