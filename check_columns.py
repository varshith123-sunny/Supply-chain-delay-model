
import pandas as pd

df = pd.read_csv("DataCoSupplyChainDataset_no_zeros.csv", encoding="latin1")
print("\n✅ COLUMN NAMES IN DATASET:\n")
for col in df.columns:
    print(col)
