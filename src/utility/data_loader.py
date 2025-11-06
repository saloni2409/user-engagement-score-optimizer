import pandas as pd
from typing import Optional

def load_parquet_data(file_path: str, engine: str = 'auto') -> Optional[pd.DataFrame]:
    """
    Loads data from a Parquet file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the Parquet file or directory.
        engine (str): The Parquet reading engine to use ('pyarrow', 'fastparquet', or 'auto').

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame containing the data, or None if an error occurs.
    """
    try:
        # pandas automatically detects columns, types, and the best engine
        df = pd.read_parquet(file_path, engine=engine)
        # print(f"✅ Successfully loaded data from: {file_path}")
        # print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except ImportError:
        print("❌ Error: Parquet engine not installed.")
        print("Please install 'pyarrow' or 'fastparquet'.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

# --- Example Usage ---
# Assuming you have a file named 'data.parquet' in the same directory
# data_df = load_parquet_data('data.parquet')

# if data_df is not None:
#     print("\nFirst 5 rows of the DataFrame:")
#     print(data_df.head())