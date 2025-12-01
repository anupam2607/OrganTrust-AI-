# ml/train_rf.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------- PATH SETTINGS ----------
# Use absolute path if running manually
DATA_PATH = Path(r"D:\projects\organtrust\data\paired_data.csv")
ARTIFACTS = Path(r"D:\projects\organtrust\ml\artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- DATA LOADING ----------
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# ---------- FEATURE SELECTION ----------
def prepare_features(df):
    # Target column: auto-detect or simulate if missing
    if "survival_1yr" not in df.columns:
        print("⚠️ 'survival_1yr' column not found, creating synthetic labels (for demo)")
        import numpy as np
        df["survival_1yr"] = (df.index % 2 == 0).astype(int)

    features = [
        "donor_age", "donor_egfr_ml_min_1_73m2", "donor_creatinine_mg_dl",
        "recipient_age", "recipient_dialysis_months", "recipient_creatinine_mg_dl_pre_tx",
        "recipient_hemoglobin_g_dl", "recipient_hla_antibodies_count"
    ]
    for f in features:
        if f not in df.columns:
            print(f"⚠️ Feature missing: {f}, filling with 0s")
            df[f] = 0

    X = df[features].fillna(df[features].median())
    y = df["survival_1yr"]
    return X, y

# ---------- MODEL TRAINING ----------
def train():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print("\n📊 Evaluation Results:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, ARTIFACTS / "rf_model.pkl")
    joblib.dump(scaler, ARTIFACTS / "scaler.pkl")

    print(f"\n✅ Model and Scaler saved to: {ARTIFACTS}")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    train()
