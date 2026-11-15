"""
EchoPM — Dual-Stage ML Model Trainer (PS 03)
Trains:
1. Unsupervised IsolationForest (Healthy Baseline Anomaly Detector -> 0-100 Health Score)
2. Supervised RandomForestClassifier (8-Class Fault Classifier with Confidence Gate)
3. Z-Score Domain Adaptation Scaler (Matches Public/Simulated Data to Local Healthy Rig)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

np.random.seed(42)

FEATURE_COLS = [
    "vibration_rms",
    "fft_peak_hz",
    "temperature",
    "current_rms",
    "current_imbalance_pct",
    "voltage_dev_pct",
    "water_flow_lpm",
    "outlet_pressure_bar",
    "specific_energy_whl",
    "phase_mode"
]

FAULT_CLASSES = [
    "Healthy",
    "Drive-end Bearing Wear",
    "Shaft Misalignment / Loose Mount",
    "External Voltage Sag / Phase Imbalance",
    "Dry Running / Inlet Blockage",
    "Impeller Wear / Cavitation",
    "Stator Winding / Overcurrent Fault",
    "Capacitor Degradation (1-Phase)"
]

def generate_synthetic_dataset(samples_per_class=350):
    rows = []

    for fault in FAULT_CLASSES:
        for _ in range(samples_per_class):
            phase_mode = np.random.choice([1, 3])
            
            # Base healthy parameters (aligned to 12V DC / 1P & 3P pump rig)
            vib = np.random.normal(1.15, 0.18)
            fft = np.random.normal(48.0, 4.0)
            temp = np.random.normal(42.5, 2.0)
            curr = np.random.normal(3.40, 0.20)
            imb = np.random.normal(1.2, 0.4) if phase_mode == 3 else 0.0
            vdev = np.random.normal(1.0, 0.5)
            flow = np.random.normal(18.5, 0.8)
            press = np.random.normal(1.15, 0.06)

            if fault == "Healthy":
                pass
            elif fault == "Drive-end Bearing Wear":
                vib = np.random.normal(2.55, 0.30)
                fft = np.random.normal(165.0, 18.0)
                temp = np.random.normal(50.5, 2.5)
                curr = np.random.normal(4.15, 0.25)
                flow = np.random.normal(17.8, 0.9)
            elif fault == "Shaft Misalignment / Loose Mount":
                vib = np.random.normal(2.90, 0.35)
                fft = np.random.normal(100.0, 8.0)
                temp = np.random.normal(46.0, 2.0)
                curr = np.random.normal(3.85, 0.22)
            elif fault == "External Voltage Sag / Phase Imbalance":
                vdev = np.random.normal(14.5, 2.5)
                imb = np.random.normal(16.0, 3.5) if phase_mode == 3 else 0.0
                curr = np.random.normal(4.30, 0.30)
                temp = np.random.normal(53.0, 3.0)
                flow = np.random.normal(15.2, 1.1)
                press = np.random.normal(0.92, 0.08)
            elif fault == "Dry Running / Inlet Blockage":
                curr = np.random.normal(1.65, 0.25)
                flow = np.random.normal(2.2, 1.0)
                press = np.random.normal(0.18, 0.07)
                temp = np.random.normal(56.0, 3.5)
                vib = np.random.normal(2.10, 0.28)
            elif fault == "Impeller Wear / Cavitation":
                vib = np.random.normal(2.35, 0.25)
                fft = np.random.normal(240.0, 25.0)
                flow = np.random.normal(10.5, 1.4)
                press = np.random.normal(0.58, 0.09)
                curr = np.random.normal(2.95, 0.25)
            elif fault == "Stator Winding / Overcurrent Fault":
                curr = np.random.normal(5.35, 0.40)
                temp = np.random.normal(64.0, 4.0)
                imb = np.random.normal(12.5, 2.5) if phase_mode == 3 else 0.0
                vib = np.random.normal(1.85, 0.25)
            elif fault == "Capacitor Degradation (1-Phase)":
                phase_mode = 1
                imb = 0.0
                curr = np.random.normal(4.65, 0.35)
                temp = np.random.normal(55.5, 3.0)
                vib = np.random.normal(2.05, 0.25)
                flow = np.random.normal(13.8, 1.2)

            power_kw = round((curr * 230.0) / 1000.0, 3)
            spec_energy = round((power_kw * 1000.0) / max(flow * 60.0, 1.0), 3)

            rows.append({
                "vibration_rms": round(max(vib, 0.1), 3),
                "fft_peak_hz": round(max(fft, 10.0), 2),
                "temperature": round(max(temp, 20.0), 2),
                "current_rms": round(max(curr, 0.2), 3),
                "current_imbalance_pct": round(max(imb, 0.0), 2),
                "voltage_dev_pct": round(max(vdev, 0.0), 2),
                "water_flow_lpm": round(max(flow, 0.0), 2),
                "outlet_pressure_bar": round(max(press, 0.0), 3),
                "specific_energy_whl": spec_energy,
                "phase_mode": int(phase_mode),
                "fault_label": fault
            })

    return pd.DataFrame(rows)

def train_and_save():
    os.makedirs("artifacts", exist_ok=True)
    df = generate_synthetic_dataset(samples_per_class=400)
    df.to_csv("artifacts/master_training_dataset.csv", index=False)

    X = df[FEATURE_COLS]
    y = df["fault_label"]

    # Healthy-only subset for Baseline Calibration & Isolation Forest
    healthy_df = df[df["fault_label"] == "Healthy"]
    baseline_stats = {}
    for col in FEATURE_COLS:
        baseline_stats[col] = {
            "mean": float(healthy_df[col].mean()),
            "std": float(max(healthy_df[col].std(), 1e-4))
        }
    with open("artifacts/baseline_profile.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_healthy_scaled = scaler.transform(healthy_df[FEATURE_COLS])

    # Stage 1: Unsupervised Anomaly Detector
    iso_forest = IsolationForest(n_estimators=150, contamination=0.04, random_state=42)
    iso_forest.fit(X_healthy_scaled)

    # Stage 2: Supervised Fault Classifier
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    print("=== EchoPM Hybrid ML Validation Report ===")
    print(classification_report(y_test, y_pred))

    joblib.dump(scaler, "artifacts/feature_scaler.pkl")
    joblib.dump(iso_forest, "artifacts/isolation_forest.pkl")
    joblib.dump(rf_clf, "artifacts/fault_classifier.pkl")
    print("Saved ML artifacts to ml/artifacts/")

if __name__ == "__main__":
    train_and_save()
