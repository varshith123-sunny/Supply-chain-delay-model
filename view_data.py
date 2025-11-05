import pandas as pd

# Path to your dataset
file_path = "data/DataCoSupplyChainDataset_no_zeros.csv"

# Read the dataset
df = pd.read_csv(file_path)

# Show basic info
print("\n✅ Dataset Loaded Successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# Show first few rows
print("\nPreview of Data:")
print(df.head())

# Optional: Show column names
print("\nColumn Names:")
print(df.columns.tolist())