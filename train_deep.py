# ml/train_deep.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# ---------- PATH SETTINGS ----------
DATA_PATH = Path(r"D:\projects\organtrust\data\paired_data.csv")
ARTIFACTS = Path(r"D:\projects\organtrust\ml\artifacts")
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# ---------- LOAD + PREPARE DATA ----------
def load_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target column check or synthetic fallback
    if "survival_1yr" not in df.columns:
        print("⚠️ 'survival_1yr' column not found, creating synthetic labels (for demo)")
        df["survival_1yr"] = (df.index % 2 == 0).astype(int)

    features = [
        "donor_age", "donor_egfr_ml_min_1_73m2", "donor_creatinine_mg_dl",
        "recipient_age", "recipient_dialysis_months", "recipient_creatinine_mg_dl_pre_tx",
        "recipient_hemoglobin_g_dl", "recipient_hla_antibodies_count"
    ]

    # Fill any missing columns
    for f in features:
        if f not in df.columns:
            print(f"⚠️ Missing feature: {f}, filling with zeros")
            df[f] = 0

    X = df[features].fillna(df[features].median())
    y = df["survival_1yr"].astype(int)
    return X, y

# ---------- BUILD MODEL ----------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------- TRAIN FUNCTION ----------
def train():
    X, y = load_prepare()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Build and train model
    model = build_model(X_train_s.shape[1])
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True
    )

    print("\n🚀 Starting training...\n")
    history = model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[callback],
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print(f"✅ Test Loss: {loss:.4f}")

    # Save model + scaler
    model.save(ARTIFACTS / "deep_model.h5")
    joblib.dump(scaler, ARTIFACTS / "deep_scaler.pkl")
    print(f"\n💾 Saved model and scaler to {ARTIFACTS}")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    train()
