# EchoPM — Predictive Maintenance & Utility Dashboard (PS 03)

Complete, runnable end-to-end software package featuring:
- **Dual-Stage Hybrid ML (`ml/train_model.py`):** Isolation Forest anomaly detector + Random Forest 8-class fault classifier with a 65% confidence Unknown Fault gate.
- **Flask Inference, Runway & BER Cost Server (`backend/app.py`):** Live MQTT subscriber, built-in fault-injection simulator, Safe Operating Runway estimator, and Indian market Spare Parts + Tiered Labour BER feasibility calculator.
- **Dark-Mode Industrial Dashboard (`frontend/dashboard.py`):** Matches the EchoPM UI with 1-Phase/3-Phase toggle, Novelty Cards, 6 IoT telemetry gauges, repair checklist, unknown fault fallback banner, and dual-axis trend charts.

## Quick Start (3 Steps)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Python 3.12 is recommended for the pinned NumPy and scikit-learn versions.

2. **Start the ML + Flask Backend (Terminal 1):**
   ```bash
   python backend/app.py
   ```
   *(Defaults to API port 5001; set `ECHOPM_PORT` in both terminals to use another port. Automatically trains and saves the `.pkl` ML models on first launch and starts streaming telemetry.)*

3. **Launch the Streamlit Dashboard (Terminal 2):**
   ```bash
   streamlit run frontend/dashboard.py
   ```
