import pandas as pd


def load_iris_dataset():
    """Load the Iris dataset"""
    return pd.read_csv("data/raw/classification/iris/iris.csv")

def load_wine_dataset():
    """Load the Wine dataset"""
    return pd.read_csv("data/raw/classification/wine/wine.csv")

def load_california_housing():
    """Load the California Housing dataset"""
    return pd.read_csv("data/raw/regression/housing/california_housing.csv")
