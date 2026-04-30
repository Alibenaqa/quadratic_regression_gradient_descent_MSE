import numpy as np
import pandas as pd


def load_data(path):
    # Load CSV file
    df = pd.read_csv(path, index_col=0)
    print(df.head())
    print(df.dtypes)
    print(f"Nombre de lignes : {len(df)}")
    return df["surface"].values, df["prix"].values


def standardize(arr):
    # Center and scale
    mu, sigma = arr.mean(), arr.std()
    return (arr - mu) / sigma, mu, sigma
