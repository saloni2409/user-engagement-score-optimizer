import pandas as pd
from pathlib import Path

# Define the path to your Parquet file
data_path = Path().absolute().parent / 'data' / 'breuninger_user_product_event_counts_000000000000'
parquet_file_path = data_path

try:
    # Use pandas to read the Parquet file into a DataFrame
    # pandas will automatically use 'pyarrow' or 'fastparquet' if available
    df = pd.read_parquet(parquet_file_path)

    print("✅ Parquet file read successfully into a Pandas DataFrame.")
    print("\n--- DataFrame Head ---")
    print(df.head())
    print("\n--- DataFrame Info ---")
    df.info()

except FileNotFoundError:
    print(f"❌ Error: The file '{parquet_file_path}' was not found.")
except Exception as e:
    print(f"❌ An error occurred while reading the Parquet file: {e}")