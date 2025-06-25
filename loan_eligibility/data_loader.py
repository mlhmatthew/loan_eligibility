import pandas as pd

def load_data(path):
    """Loads CSV data from the given path."""
    return pd.read_csv(path)
