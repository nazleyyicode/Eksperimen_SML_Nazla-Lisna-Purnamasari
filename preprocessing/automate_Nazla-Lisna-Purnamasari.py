import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def preprocess_weather_data(input_path, output_path, target_column='Rain'):
    # membaca dataset
    df = pd.read_csv(input_path)

    # menangani missing value dan duplikat
    df = df.dropna()
    df = df.drop_duplicates()

    # encoding
    encoder = LabelEncoder()
    df[target_column] = encoder.fit_transform(df[target_column])

    # scaling
    X = df.drop(columns=target_column)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed[target_column] = y.values
    
    #menyimpan hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)

    print("Preprocessing selesai. Dataset tersimpan.")

if __name__ == "__main__":
    input_path = "namadataset_raw/weather_forecast.csv"
    output_path = "preprocessing/namadataset_preprocessing/weather_forecast_preprocessing.csv"
    preprocess_weather_data(input_path, output_path)