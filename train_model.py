"""
train_model.py
==============
Trains a Random Forest Classifier on the Crop Recommendation dataset
using all seven soil and weather features.

Features used (in order):
1. N (Nitrogen)
2. P (Phosphorus)
3. K (Potassium)
4. temperature (Celsius)
5. humidity (Percentage)
6. ph (Soil pH)
7. rainfall (Millimeters)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURE_NAMES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data', 'crop_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')

def train():
    print("=" * 60)
    print("   BASIC CROP RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run generate_data.py first.")

    print(f"\n[1/5] Loading crop dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"      Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"      Available Features: {', '.join(FEATURE_NAMES)}")

    # Verify required columns exist
    missing_cols = [col for col in FEATURE_NAMES + ['label'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Inspect unique classes
    classes = sorted(df['label'].unique())
    print(f"      Target Crops ({len(classes)} classes): {', '.join(classes[:8])}... and more.")

    # 2. Prepare Features & Target
    X = df[FEATURE_NAMES]
    y = df['label']

    # 3. Train / Test Split
    print("\n[2/5] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 4. Model Initialization & Training
    print("\n[3/5] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=1
    )
    rf_model.fit(X_train, y_train)

    # 5. Evaluate Model
    print("\n[4/5] Evaluating Model Performance...")
    y_pred = rf_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"      Validation Accuracy: {test_accuracy * 100:.2f}%")
    print("      (Note: Accuracy is computed on the hold-out validation split of this dataset.)")

    print("\nClassification Report (Sample Summary):")
    report = classification_report(y_test, y_pred, output_dict=True)
    for crop in classes[:5]:
        p = report[crop]['precision']
        r = report[crop]['recall']
        f1 = report[crop]['f1-score']
        print(f"      - {crop:12s}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

    # 6. Save Model Artifact
    print(f"\n[5/5] Saving model artifact...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        'model': rf_model,
        'features': FEATURE_NAMES,
        'classes': classes,
        'test_accuracy': float(test_accuracy)
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(artifact, f)

    print(f"      Saved model to: {MODEL_PATH}")

    # Also save at root directory as fallback for convenience
    root_model_path = os.path.join(os.path.dirname(__file__), 'crop_model.pkl')
    with open(root_model_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"      Saved copy to: {root_model_path}")

    # 7. Verification sanity check
    print("\nSanity Check (Sample Prediction):")
    sample_input = pd.DataFrame([{
        'N': 80, 'P': 40, 'K': 40,
        'temperature': 24.0, 'humidity': 82.0,
        'ph': 6.5, 'rainfall': 240.0
    }], columns=FEATURE_NAMES)
    predicted_crop = rf_model.predict(sample_input)[0]
    print(f"      Input Rice Conditions -> Predicted: '{predicted_crop}'")
    print("=" * 60)
    print("Training finished successfully! You can now start 'app.py'.")
    print("=" * 60)

if __name__ == '__main__':
    train()
