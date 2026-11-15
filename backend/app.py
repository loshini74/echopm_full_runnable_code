"""
EchoPM — Backend Inference, Calibration, Runway & BER Cost Engine (PS 03)
- Serves REST APIs for Streamlit Dashboard
- Runs Dual-Stage ML (Isolation Forest + Random Forest) with <65% Unknown Fault Gate
- Calculates Safe Operating Runway (Days) & BER Repair Budget (Spare Parts + Tiered Labour)
- Supports Live ESP32 MQTT Ingestion AND Built-in Fault Injection Simulator for Demos
"""

import os
import sys
import json
import hmac
import time
import sqlite3
import threading
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt

# Ensure ML artifacts exist before booting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "ml"))
ARTIFACTS_DIR = os.path.join(ML_DIR, "artifacts")
DB_PATH = os.path.join(BASE_DIR, "echopm_plant.db")

if not os.path.exists(os.path.join(ARTIFACTS_DIR, "fault_classifier.pkl")):
    sys.path.insert(0, ML_DIR)
    import train_model
    orig_cwd = os.getcwd()
    os.chdir(ML_DIR)
    train_model.train_and_save()
    os.chdir(orig_cwd)

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_scaler.pkl"))
iso_forest = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl"))
rf_clf = joblib.load(os.path.join(ARTIFACTS_DIR, "fault_classifier.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "baseline_profile.json"), "r") as f:
    BASELINE_PROFILE = json.load(f)

FEATURE_COLS = [
    "vibration_rms", "fft_peak_hz", "temperature", "current_rms",
    "current_imbalance_pct", "voltage_dev_pct", "water_flow_lpm",
    "outlet_pressure_bar", "specific_energy_whl", "phase_mode"
]

NEW_MOTOR_REPLACEMENT_COST = 4500  # INR benchmark for 1HP / Closed-Loop Pump Motor
API_PORT = int(os.environ.get("PORT") or os.environ.get("ECHOPM_PORT", "5001"))
API_KEY = os.environ.get("ECHOPM_API_KEY")

# Realistic Indian Market Cost & Guided Repair Lookup (Spare Parts + Tiered Labour)
REPAIR_KNOWLEDGE_BASE = {
    "Healthy": {
        "status_badge": "HEALTHY (Nominal Operation)",
        "runway_days": 180,
        "budget_range": "₹0 (INR)",
        "spare_parts": "None required (₹0)",
        "labour_cost": "Routine operator check (₹0)",
        "ber_ratio_pct": 0,
        "ber_verdict": "OPTIMAL (No Action Needed)",
        "delay_penalty": "₹0/day (Baseline Efficiency)",
        "checklist": [
            "1. Continue automated condition monitoring.",
            "2. Verify inlet strainer cleanliness weekly.",
            "3. Log specific energy (Wh/L) trend."
        ]
    },
    "Drive-end Bearing Wear": {
        "status_badge": "DEGRADED (Drive-end Bearing Wear)",
        "runway_days": 12,
        "budget_range": "₹650 – ₹900 (INR)",
        "spare_parts": "₹250 – ₹400 (6203-2RS Bearing + Grease)",
        "labour_cost": "₹400 – ₹500 (Field Mechanic Fitting)",
        "ber_ratio_pct": 18,
        "ber_verdict": "HIGHLY FEASIBLE (18% of New Cost)",
        "delay_penalty": "+₹180/day (Friction Energy Waste)",
        "checklist": [
            "1. Inspect drive-end bearing; check lubrication & fit.",
            "2. Verify alignment and tighten loose mounting.",
            "3. If unresolved, inspect impeller for debris."
        ]
    },
    "Shaft Misalignment / Loose Mount": {
        "status_badge": "WARNING (Shaft Misalignment / Loose Mount)",
        "runway_days": 18,
        "budget_range": "₹150 – ₹300 (INR)",
        "spare_parts": "₹0 – ₹50 (SS Shims / Mounting Bolts)",
        "labour_cost": "₹150 – ₹250 (In-House Technician)",
        "ber_ratio_pct": 6,
        "ber_verdict": "HIGHLY FEASIBLE (6% of New Cost)",
        "delay_penalty": "+₹110/day (Vibration & Seal Stress)",
        "checklist": [
            "1. Isolate supply and tighten baseplate mounting bolts.",
            "2. Re-align pump-motor coupling with dial gauge/shims.",
            "3. Re-run 3-second PWM step test to confirm."
        ]
    },
    "External Voltage Sag / Phase Imbalance": {
        "status_badge": "SUPPLY ALERT (External Voltage / Phase Fault)",
        "runway_days": 7,
        "budget_range": "₹200 – ₹450 (INR)",
        "spare_parts": "₹0 – ₹150 (Terminal Lugs / HRC Fuse)",
        "labour_cost": "₹200 – ₹300 (Plant Electrician Check)",
        "ber_ratio_pct": 8,
        "ber_verdict": "EXTERNAL SUPPLY ISSUE (Do Not Dismantle Motor)",
        "delay_penalty": "+₹210/day (Winding Thermal Stress)",
        "checklist": [
            "1. Check incoming grid voltage and starter contactor terminals.",
            "2. Inspect phase fuses and neutral/earth continuity.",
            "3. Confirm supply balance before restarting motor."
        ]
    },
    "Dry Running / Inlet Blockage": {
        "status_badge": "CRITICAL HYDRAULIC (Dry Running / Inlet Blocked)",
        "runway_days": 2,
        "budget_range": "₹150 – ₹250 (INR)",
        "spare_parts": "₹0 (Unless mechanical seal burnt)",
        "labour_cost": "₹150 – ₹250 (In-House Priming & Cleaning)",
        "ber_ratio_pct": 5,
        "ber_verdict": "IMMEDIATE ACTION (Prevent Seal Burnout)",
        "delay_penalty": "+₹260/day (Severe Overheating Risk)",
        "checklist": [
            "1. Stop pump immediately to protect mechanical seal.",
            "2. Clear suction foot-valve debris and check sump level.",
            "3. Prime pump casing and bleed trapped air."
        ]
    },
    "Impeller Wear / Cavitation": {
        "status_badge": "DEGRADED (Impeller Wear / Cavitation)",
        "runway_days": 15,
        "budget_range": "₹850 – ₹1,300 (INR)",
        "spare_parts": "₹400 – ₹700 (Brass/PPO Impeller + Seal)",
        "labour_cost": "₹450 – ₹600 (Volute Casing Teardown)",
        "ber_ratio_pct": 24,
        "ber_verdict": "FEASIBLE (24% of New Cost)",
        "delay_penalty": "+₹195/day (High Specific Energy Wh/L)",
        "checklist": [
            "1. Check suction head and inlet valve for cavitation bubbles.",
            "2. Open pump volute and inspect impeller vanes for erosion.",
            "3. Replace worn impeller and mechanical shaft seal."
        ]
    },
    "Stator Winding / Overcurrent Fault": {
        "status_badge": "CRITICAL ELECTRICAL (Stator Winding Degradation)",
        "runway_days": 3,
        "budget_range": "₹2,400 – ₹3,100 (INR)",
        "spare_parts": "₹1,100 – ₹1,500 (Copper Winding + Varnish)",
        "labour_cost": "₹1,300 – ₹1,600 (External Rewinding Workshop)",
        "ber_ratio_pct": 64,
        "ber_verdict": "NOT FEASIBLE — BER > 60% (Replace Motor)",
        "delay_penalty": "+₹310/day (Imminent Burnout Risk)",
        "checklist": [
            "1. Perform Megger insulation resistance test on windings.",
            "2. Repair exceeds 60% BER threshold; procure replacement motor.",
            "3. Salvage usable bearings and housing for spares."
        ]
    },
    "Capacitor Degradation (1-Phase)": {
        "status_badge": "DEGRADED (1-Phase Start/Run Capacitor Weak)",
        "runway_days": 10,
        "budget_range": "₹300 – ₹450 (INR)",
        "spare_parts": "₹150 – ₹250 (Oil-Filled Run Capacitor)",
        "labour_cost": "₹150 – ₹200 (Plant Electrician Swap)",
        "ber_ratio_pct": 9,
        "ber_verdict": "HIGHLY FEASIBLE (9% of New Cost)",
        "delay_penalty": "+₹145/day (High Aux Winding Current)",
        "checklist": [
            "1. Discharge capacitor safely and test µF value with multimeter.",
            "2. Replace weak capacitor with exact voltage/µF rating.",
            "3. Verify centrifugal switch release after startup."
        ]
    },
    "Unknown Fault": {
        "status_badge": "UNKNOWN FAULT (Unrecognized Signature)",
        "runway_days": 4,
        "budget_range": "Workshop Inspection Required",
        "spare_parts": "Pending manual teardown diagnosis",
        "labour_cost": "₹400 – ₹700 (Workshop Diagnostic Fee)",
        "ber_ratio_pct": 0,
        "ber_verdict": "MANUAL OVERHAUL REQUIRED",
        "delay_penalty": "+₹200/day (Unclassified Anomaly)",
        "checklist": [
            "1. Anomaly detected outside trained fault signatures (<65% confidence).",
            "2. Isolate power supply and route motor to workshop.",
            "3. Log verified workshop findings to retrain EchoPM dataset."
        ]
    }
}

app = Flask(__name__)
CORS(app)

@app.before_request
def authenticate_api_request():
    if request.path.startswith("/api/v1/") and API_KEY:
        supplied_key = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied_key, API_KEY):
            return jsonify({"error": "Unauthorized"}), 401

# Shared Live State
SYSTEM_STATE = {
    "phase_mode": 1,
    "model_mode": "Hybrid ML",
    "calibration_pct": 100,
    "active_sim_fault": "Drive-end Bearing Wear",
    "pwm_probing_active": False
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            device_id TEXT,
            phase_mode INTEGER,
            vibration_rms REAL,
            fft_peak_hz REAL,
            temperature REAL,
            current_rms REAL,
            current_imbalance_pct REAL,
            voltage_dev_pct REAL,
            power_kw REAL,
            water_flow_lpm REAL,
            outlet_pressure_bar REAL,
            specific_energy_whl REAL,
            health_score REAL,
            predicted_fault TEXT,
            confidence_pct REAL,
            is_unknown_fault INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def run_ml_inference(reading: dict):
    row_df = pd.DataFrame([{col: reading[col] for col in FEATURE_COLS}])
    X_scaled = scaler.transform(row_df)

    # 1. Unsupervised Isolation Forest -> Health Score (0-100)
    raw_score = float(iso_forest.decision_function(X_scaled)[0])
    # Map decision_function roughly [-0.25, +0.15] -> [15, 99]
    health_score = float(np.clip(72.0 + (raw_score * 180.0), 12.0, 99.0))

    # 2. Supervised Random Forest -> Fault Class & Probability
    probs = rf_clf.predict_proba(X_scaled)[0]
    classes = rf_clf.classes_
    best_idx = int(np.argmax(probs))
    best_class = str(classes[best_idx])
    best_conf = round(float(probs[best_idx]) * 100.0, 1)

    # Force Unknown Fault gate if user injected an unrecognized signature or confidence < 65%
    is_unknown = False
    if reading.get("force_unknown", False) or (best_class != "Healthy" and best_conf < 65.0):
        best_class = "Unknown Fault"
        is_unknown = True
        health_score = min(health_score, 48.0)

    if best_class == "Healthy":
        health_score = max(health_score, 88.0)
    else:
        health_score = min(health_score, 68.0)

    return round(health_score, 1), best_class, best_conf, is_unknown

def ingest_reading(reading: dict):
    health_score, pred_fault, conf_pct, is_unknown = run_ml_inference(reading)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO telemetry_logs (
            timestamp, device_id, phase_mode, vibration_rms, fft_peak_hz,
            temperature, current_rms, current_imbalance_pct, voltage_dev_pct,
            power_kw, water_flow_lpm, outlet_pressure_bar, specific_energy_whl,
            health_score, predicted_fault, confidence_pct, is_unknown_fault
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts, reading.get("device_id", "motor_pump_01"), int(reading["phase_mode"]),
        reading["vibration_rms"], reading["fft_peak_hz"], reading["temperature"],
        reading["current_rms"], reading["current_imbalance_pct"], reading["voltage_dev_pct"],
        reading["power_kw"], reading["water_flow_lpm"], reading["outlet_pressure_bar"],
        reading["specific_energy_whl"], health_score, pred_fault, conf_pct, int(is_unknown)
    ))
    conn.commit()
    conn.close()

def generate_simulated_packet(fault_name: str, phase_mode: int):
    vib, fft, temp, curr = 1.15, 48.0, 42.5, 3.40
    imb = 1.2 if phase_mode == 3 else 0.0
    vdev, flow, press = 1.0, 18.5, 1.15
    force_unknown = False

    if fault_name == "Drive-end Bearing Wear":
        vib, fft, temp, curr, flow = 2.45, 168.0, 49.3, 4.15, 18.2
        press = 1.10
    elif fault_name == "Shaft Misalignment / Loose Mount":
        vib, fft, temp, curr = 2.88, 100.0, 46.2, 3.85
    elif fault_name == "External Voltage Sag / Phase Imbalance":
        vdev, curr, temp, flow, press = 15.2, 4.35, 53.4, 15.0, 0.90
        imb = 17.5 if phase_mode == 3 else 0.0
    elif fault_name == "Dry Running / Inlet Blockage":
        curr, flow, press, temp, vib = 1.60, 1.8, 0.15, 57.2, 2.15
    elif fault_name == "Impeller Wear / Cavitation":
        vib, fft, flow, press, curr = 2.38, 245.0, 10.2, 0.55, 2.95
    elif fault_name == "Stator Winding / Overcurrent Fault":
        curr, temp, vib = 5.45, 65.0, 1.90
        imb = 13.0 if phase_mode == 3 else 0.0
    elif fault_name == "Capacitor Degradation (1-Phase)":
        curr, temp, vib, flow = 4.70, 56.0, 2.08, 13.5
    elif fault_name == "Unknown Fault (Workshop Fallback Demo)":
        # Exotic multi-axis oscillation outside trained classes
        vib, fft, temp, curr, flow, press = 3.65, 412.0, 39.0, 2.10, 19.8, 1.48
        force_unknown = True

    # Add realistic sensor jitter
    vib = round(max(0.1, vib + np.random.normal(0, 0.04)), 2)
    temp = round(max(20.0, temp + np.random.normal(0, 0.2)), 1)
    curr = round(max(0.2, curr + np.random.normal(0, 0.05)), 2)
    flow = round(max(0.0, flow + np.random.normal(0, 0.15)), 1)
    press = round(max(0.0, press + np.random.normal(0, 0.02)), 2)
    power_kw = round((curr * 230.0) / 1000.0, 2)
    spec_energy = round((power_kw * 1000.0) / max(flow * 60.0, 1.0), 2)

    return {
        "device_id": "motor_pump_01",
        "phase_mode": phase_mode,
        "vibration_rms": vib,
        "fft_peak_hz": round(fft + np.random.normal(0, 2.0), 1),
        "temperature": temp,
        "current_rms": curr,
        "current_imbalance_pct": round(imb, 1),
        "voltage_dev_pct": round(vdev, 1),
        "power_kw": power_kw,
        "water_flow_lpm": flow,
        "outlet_pressure_bar": press,
        "specific_energy_whl": spec_energy,
        "force_unknown": force_unknown
    }

# Background Simulator Loop (publishes live telemetry every 2.5s when hardware isn't streaming)
def background_simulator():
    # Pre-seed 25 historical points so charts look rich immediately
    for i in range(25):
        pkt = generate_simulated_packet(SYSTEM_STATE["active_sim_fault"], SYSTEM_STATE["phase_mode"])
        ingest_reading(pkt)
    while True:
        pkt = generate_simulated_packet(SYSTEM_STATE["active_sim_fault"], SYSTEM_STATE["phase_mode"])
        ingest_reading(pkt)
        time.sleep(2.5)

# MQTT Listener for Live ESP32 Hardware Node
def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    client.subscribe("factory/line1/+/telemetry")

def on_mqtt_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8"))
        data.setdefault("phase_mode", SYSTEM_STATE["phase_mode"])
        data.setdefault("fft_peak_hz", 160.0 if data.get("vibration_rms", 1.0) > 2.2 else 50.0)
        data.setdefault("current_imbalance_pct", 0.0)
        data.setdefault("voltage_dev_pct", 1.0)
        data.setdefault("water_flow_lpm", 18.0)
        data.setdefault("outlet_pressure_bar", 1.1)
        pwr = data.get("power_kw", (data.get("current_rms", 3.5) * 230.0) / 1000.0)
        data["power_kw"] = pwr
        data["specific_energy_whl"] = round((pwr * 1000.0) / max(data["water_flow_lpm"] * 60.0, 1.0), 2)
        ingest_reading(data)
    except Exception as e:
        print(f"MQTT parse error: {e}")

try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.connect_async("broker.hivemq.com", 1883, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"MQTT offline fallback mode active: {e}")

# REST API Endpoints
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/api/v1/dashboard", methods=["GET"])
def get_dashboard_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 40", conn)
    conn.close()

    if df.empty:
        return jsonify({"status": "waiting"}), 404

    latest = df.iloc[0].to_dict()
    history = df.iloc[::-1].to_dict(orient="records")

    fault_key = latest["predicted_fault"]
    guidance = REPAIR_KNOWLEDGE_BASE.get(fault_key, REPAIR_KNOWLEDGE_BASE["Unknown Fault"])

    return jsonify({
        "system": SYSTEM_STATE,
        "latest": latest,
        "guidance": guidance,
        "history": history
    })

@app.route("/api/v1/control", methods=["POST"])
def update_control():
    body = request.json or {}
    if "phase_mode" in body:
        SYSTEM_STATE["phase_mode"] = int(body["phase_mode"])
    if "model_mode" in body:
        SYSTEM_STATE["model_mode"] = str(body["model_mode"])
    if "active_sim_fault" in body:
        SYSTEM_STATE["active_sim_fault"] = str(body["active_sim_fault"])
        # Immediately inject a sample of the newly selected scenario
        pkt = generate_simulated_packet(SYSTEM_STATE["active_sim_fault"], SYSTEM_STATE["phase_mode"])
        ingest_reading(pkt)
    if body.get("trigger_pwm_probe"):
        SYSTEM_STATE["pwm_probing_active"] = True
    if body.get("save_verified_window"):
        conn = sqlite3.connect(DB_PATH)
        latest_df = pd.read_sql_query("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 1", conn)
        conn.close()
        if not latest_df.empty:
            csv_path = os.path.join(ARTIFACTS_DIR, "master_training_dataset.csv")
            latest_df["fault_label"] = latest_df["predicted_fault"]
            latest_df[FEATURE_COLS + ["fault_label"]].to_csv(csv_path, mode="a", header=False, index=False)
    return jsonify({"status": "updated", "system": SYSTEM_STATE})

if __name__ == "__main__":
    sim_thread = threading.Thread(target=background_simulator, daemon=True)
    sim_thread.start()
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
