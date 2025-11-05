import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ✅ Load dataset
data = pd.read_csv("data/DataCoSupplyChainDataset_no_zeros.csv", encoding="latin1")

# ✅ If Late_delivery_risk column does not exist or contains only one class, recreate it
if "Late_delivery_risk" not in data.columns or data["Late_delivery_risk"].nunique() < 2:
    print("⚠  Recreating Late_delivery_risk column based on shipping delay...")
    data["Late_delivery_risk"] = (
        data["Days for shipping (real)"] > data["Days for shipment (scheduled)"]
    ).astype(int)

# ✅ Print class distribution
print("\n📊 Class distribution:")
print(data["Late_delivery_risk"].value_counts())

# ✅ Select working features that exist in dataset
X = data[[
    "Order City", "Order State", "Category Name",
    "Order Item Quantity", "Order Item Discount", "Product Price"
]]

y = data["Late_delivery_risk"]

# ✅ Columns by type
categorical_cols = ["Order City", "Order State", "Category Name"]
numerical_cols = ["Order Item Quantity", "Order Item Discount", "Product Price"]

# ✅ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols)
    ]
)

# ✅ Full pipeline (Preprocess + Model)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train model
pipeline.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = pipeline.predict(X_test)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
os.makedirs("saved_objects", exist_ok=True)
joblib.dump(pipeline, "saved_objects/pipeline.pkl")
print("\n📁 Model saved at: saved_objects/pipeline.pkl")