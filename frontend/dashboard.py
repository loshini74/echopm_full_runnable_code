"""
EchoPM — Predictive Maintenance & Utility Dashboard (PS 03 | iTech IDEAthon 2027)
Matches the exact dark-mode industrial UI mockup:
- Top Header Bar with Asset ID & 100% Adaptive Calibration Badge
- Left Sidebar: System Config, Model Selector, Phase Mode (1-Phase / 3-Phase), Calibration Progress, Signal Quick-Toggles & Live Fault Simulator
- Row 1 (Novelty Cards): Asset Condition Card, Safe Operating Runway Card, Estimated Repair Budget (Spare Parts + Tiered Labour + BER Feasibility + Delay Penalty)
- Row 2 (Real-time Telemetry): 6 Color-Coded Gauge Cards with Sparklines (Vibration RMS, Temperature, Motor Current, Power Consumption, Water Flow, Outlet Pressure)
- Row 3 (AI Diagnostics & Repair Guidance): Recommended Way of Repair Checklist, Unknown Fault Workshop Fallback Alert, and Dual-Axis Vibration vs. Power Consumption Chart
"""

import os
import time
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="EchoPM — Predictive Maintenance & Utility Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_PORT = os.environ.get("ECHOPM_PORT", "5001")
API_BASE = f"http://localhost:{API_PORT}/api/v1"

# Custom Dark-Mode Industrial Styling matching the mockup
st.markdown("""
<style>
    .stApp {
        background-color: #121518;
        color: #E6EDF3;
    }
    section[data-testid="stSidebar"] {
        background-color: #181C20;
        border-right: 1px solid #2D333B;
    }
    .top-banner {
        background-color: #1B2026;
        border: 1px solid #2D333B;
        border-radius: 10px;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    .ps-badge {
        background-color: #3D3522;
        color: #F2CC60;
        border: 1px solid #6E5A28;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 12px;
    }
    .cal-badge {
        background-color: #1B4332;
        color: #52B788;
        border: 1px solid #2D6A4F;
        padding: 5px 14px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .novelty-card {
        background-color: #1B2026;
        border: 1px solid #2D333B;
        border-radius: 10px;
        padding: 16px 20px;
        height: 215px;
    }
    .novelty-card-warn {
        background: linear-gradient(135deg, #3A1D1D 0%, #1B2026 75%);
        border: 1px solid #D9534F;
        border-left: 5px solid #E05252;
        border-radius: 10px;
        padding: 16px 20px;
        height: 215px;
    }
    .novelty-card-ok {
        background: linear-gradient(135deg, #163325 0%, #1B2026 75%);
        border: 1px solid #2D6A4F;
        border-left: 5px solid #40C057;
        border-radius: 10px;
        padding: 16px 20px;
        height: 215px;
    }
    .card-title {
        color: #ADB5BD;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .unknown-banner {
        background-color: #332B14;
        border: 1px solid #8A6D1B;
        color: #FFD43B;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-top: 10px;
    }
    .guidance-box {
        background-color: #1B2026;
        border: 1px solid #2D333B;
        border-radius: 10px;
        padding: 16px;
        min-height: 185px;
    }
</style>
""", unsafe_allow_html=True)

def post_control(payload):
    try:
        requests.post(f"{API_BASE}/control", json=payload, timeout=1.5)
    except Exception:
        pass

def make_gauge_card(title, value, unit, min_val, max_val, warn_thresh, status_text, status_color, spark_series):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 24, "color": "#FFFFFF"}},
        gauge={
            "axis": {"range": [min_val, max_val], "visible": False},
            "bar": {"color": status_color, "thickness": 0.28},
            "bgcolor": "#2B3038",
            "borderwidth": 0,
            "threshold": {
                "line": {"color": "#FF6B6B", "width": 3},
                "thickness": 0.75,
                "value": warn_thresh
            }
        }
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#CED4DA"}, "x": 0.5},
        paper_bgcolor="#1B2026",
        plot_bgcolor="#1B2026",
        height=155,
        margin=dict(l=14, r=14, t=36, b=8)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"<div style='text-align:center; margin-top:-18px; margin-bottom:6px; font-size:0.82rem; color:{status_color}; font-weight:600;'>● {status_text}</div>",
        unsafe_allow_html=True
    )

# Sidebar Controls matching the left panel in the dashboard image
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    model_choice = st.selectbox(
        "Model Selector",
        ["Hybrid ML (Isolation Forest + RF)", "Adaptive Threshold Engine", "CWRU Benchmark Mode"],
        index=0
    )

    st.markdown("**Phase Mode**")
    phase_choice = st.radio(
        "Phase Mode",
        ["1-Phase", "3-Phase"],
        horizontal=True,
        label_visibility="collapsed"
    )
    phase_num = 1 if phase_choice == "1-Phase" else 3

    st.markdown("**Calibration Status**")
    st.progress(100, text="100% Adaptive Baseline Set")

    st.markdown("---")
    st.markdown("**Live Fault & Demo Scenario Injector**")
    sim_fault = st.selectbox(
        "Simulate Condition / Fault",
        [
            "Drive-end Bearing Wear",
            "Healthy",
            "Shaft Misalignment / Loose Mount",
            "External Voltage Sag / Phase Imbalance",
            "Dry Running / Inlet Blockage",
            "Impeller Wear / Cavitation",
            "Stator Winding / Overcurrent Fault",
            "Capacitor Degradation (1-Phase)",
            "Unknown Fault (Workshop Fallback Demo)"
        ],
        index=0
    )

    if st.button("Apply Scenario & Phase Mode", use_container_width=True):
        post_control({
            "phase_mode": phase_num,
            "model_mode": model_choice,
            "active_sim_fault": sim_fault
        })

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("⚡ PWM Probe", use_container_width=True):
            post_control({"trigger_pwm_probe": True})
            st.toast("Sent PWM 70%->100% speed-step test command to ESP32!")
    with col_b2:
        if st.button("💾 Save Data", use_container_width=True):
            post_control({"save_verified_window": True})
            st.toast("Appended calibrated window to master_training_dataset.csv!")

    st.markdown("---")
    st.markdown("**Quick-toggle Trend Signals**")
    show_vib = st.checkbox("Vibration RMS", value=True)
    show_pwr = st.checkbox("Power Consumption", value=True)
    show_flow = st.checkbox("Water Flow", value=True)
    show_press = st.checkbox("Outlet Pressure", value=False)
    auto_refresh = st.checkbox("Live Auto-Refresh (3s)", value=True)

# Fetch live backend state
try:
    resp = requests.get(f"{API_BASE}/dashboard", timeout=2.0)
    resp.raise_for_status()
    payload = resp.json()
except requests.exceptions.JSONDecodeError as e:
    st.error(f"Backend returned invalid JSON from {API_BASE}/dashboard. Ensure the EchoPM backend is running. Details: {e}")
    st.stop()
except requests.HTTPError as e:
    st.error(f"EchoPM backend returned HTTP {e.response.status_code} from {API_BASE}/dashboard.")
    st.stop()
except requests.RequestException as e:
    st.error(f"Cannot reach EchoPM backend at {API_BASE}/dashboard. Start `python backend/app.py` first. Details: {e}")
    st.stop()

latest = payload["latest"]
guidance = payload["guidance"]
history_df = pd.DataFrame(payload["history"])

# Top Header Banner
st.markdown(f"""
<div class="top-banner">
    <div>
        <span style="font-size:1.35rem; font-weight:700; color:#FFFFFF;">📊 EchoPM — Predictive Maintenance & Utility Dashboard</span>
        <span class="ps-badge">PS 03 | iTech IDEAthon 2027</span>
    </div>
    <div>
        <span style="color:#ADB5BD; margin-right:16px; font-size:0.92rem;">
            Asset ID: <b>{latest['device_id']}</b> ({latest['phase_mode']}-Phase / Closed-Loop Pump)
        </span>
        <span class="cal-badge">100% Calibrated (Adaptive Baseline Set)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 1: 1. Machine Condition & Risk Analytics (Novelty Cards)
st.markdown("#### 1. Machine Condition & Risk Analytics (Novelty Cards)")
c1, c2, c3 = st.columns([1.15, 1.0, 1.25])

is_healthy = latest["predicted_fault"] == "Healthy"
card_css = "novelty-card-ok" if is_healthy else "novelty-card-warn"
title_color = "#51CF66" if is_healthy else "#F59F00"
conf_label = "High" if latest["confidence_pct"] >= 80 else ("Moderate" if latest["confidence_pct"] >= 65 else "Low (<65% Gate)")

with c1:
    st.markdown(f"""
    <div class="{card_css}">
        <div class="card-title">Asset Condition Card</div>
        <div style="font-size:1.35rem; font-weight:700; color:{title_color}; line-height:1.25; margin-bottom:10px;">
            ● {guidance['status_badge']}
        </div>
        <div style="font-size:0.95rem; color:#E9ECEF;">
            Health Score: <b>{latest['health_score']} / 100</b> &nbsp;|&nbsp; Classifier: <b>{latest['confidence_pct']}%</b>
        </div>
        <div style="font-size:0.9rem; color:#ADB5BD; margin-top:6px;">
            Confidence: <span style="color:#51CF66; font-weight:600;">{conf_label}</span>
            &nbsp;•&nbsp; Utility: <b>{latest['specific_energy_whl']} Wh/L</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    runway_color = "#51CF66" if is_healthy else "#F59F00"
    st.markdown(f"""
    <div class="novelty-card" style="text-align:center;">
        <div class="card-title" style="text-align:left;">Safe Operating Runway Card</div>
        <div style="font-size:1.85rem; font-weight:800; color:{runway_color}; margin-top:18px; margin-bottom:8px;">
            ⏳ ~{guidance['runway_days']} DAYS REMAINING
        </div>
        <div style="font-size:0.9rem; color:#ADB5BD;">
            Schedule maintenance before critical failure
        </div>
        <div style="font-size:0.82rem; color:#868E96; margin-top:12px;">
            Phase Imbalance: {latest['current_imbalance_pct']}% &nbsp;|&nbsp; Grid Dev: {latest['voltage_dev_pct']}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="novelty-card">
        <div class="card-title">Estimated Repair Budget (Local Market Rate)</div>
        <div style="font-size:1.75rem; font-weight:800; color:#40C057; text-align:center; margin-bottom:8px;">
            {guidance['budget_range']}
        </div>
        <ul style="font-size:0.84rem; color:#DEE2E6; margin:0; padding-left:18px; line-height:1.48;">
            <li><b>Spare Parts:</b> <span style="color:#FCC419;">{guidance['spare_parts']}</span></li>
            <li><b>Service & Labour:</b> <span style="color:#FCC419;">{guidance['labour_cost']}</span></li>
            <li><b>BER Feasibility:</b> <span style="color:#51CF66; font-weight:600;">{guidance['ber_verdict']}</span></li>
            <li><b>Delay Penalty:</b> <span style="color:#FF8787;">{guidance['delay_penalty']}</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Row 2: 2. Real-time Telemetry (IoT Signals)
st.markdown("#### 2. Real-time Telemetry (IoT Signals)")
g1, g2, g3, g4, g5, g6 = st.columns(6)

with g1:
    vib_col = "#F59F00" if latest["vibration_rms"] > 2.2 else "#40C057"
    vib_st = "WARNING (Thresh 2.5g)" if latest["vibration_rms"] > 2.2 else "Normal"
    make_gauge_card("Vibration RMS", latest["vibration_rms"], "g", 0, 4.0, 2.5, vib_st, vib_col, history_df["vibration_rms"])

with g2:
    tmp_col = "#FA5252" if latest["temperature"] > 54.0 else "#40C057"
    tmp_st = "Elevated" if latest["temperature"] > 54.0 else "Normal"
    make_gauge_card("Temperature", latest["temperature"], "°C", 20, 80, 55.0, tmp_st, tmp_col, history_df["temperature"])

with g3:
    cur_col = "#FA5252" if latest["current_rms"] > 4.0 else "#40C057"
    cur_st = "Elevated" if latest["current_rms"] > 4.0 else "Normal"
    make_gauge_card("Motor Current", latest["current_rms"], "A", 0, 7.0, 4.0, cur_st, cur_col, history_df["current_rms"])

with g4:
    make_gauge_card("Power Consumption", latest["power_kw"], "kW", 0, 2.0, 1.3, "Active Load", "#339AF0", history_df["power_kw"])

with g5:
    flw_col = "#FA5252" if latest["water_flow_lpm"] < 12.0 else "#40C057"
    flw_st = "Low Flow" if latest["water_flow_lpm"] < 12.0 else "Normal"
    make_gauge_card("Water Flow", latest["water_flow_lpm"], "L/min", 0, 25.0, 12.0, flw_st, flw_col, history_df["water_flow_lpm"])

with g6:
    prs_col = "#FA5252" if latest["outlet_pressure_bar"] < 0.7 else "#40C057"
    prs_st = "Low Pressure" if latest["outlet_pressure_bar"] < 0.7 else "Normal"
    make_gauge_card("Outlet Pressure", latest["outlet_pressure_bar"], "Bar", 0, 2.0, 0.7, prs_st, prs_col, history_df["outlet_pressure_bar"])

# Row 3: AI Diagnostics & Repair Guidance
st.markdown("#### AI Diagnostics & Repair Guidance")
b1, b2, b3 = st.columns([1.15, 1.0, 1.25])

with b1:
    steps_html = "<br>".join(guidance["checklist"])
    st.markdown(f"""
    <div class="guidance-box">
        <div class="card-title">Recommended Way of Repair</div>
        <div style="font-size:0.92rem; color:#E9ECEF; line-height:1.65;">
            {steps_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

with b2:
    is_unk = bool(latest["is_unknown_fault"])
    unk_bg = "#4A1C1C" if is_unk else "#1B2026"
    unk_border = "#FA5252" if is_unk else "#2D333B"
    unk_msg = (
        "🚨 <b>ACTIVE FALLBACK TRIGGERED:</b> Fault unrecognized with the current dataset (<65% confidence). Please route motor to workshop for manual overhaul."
        if is_unk else
        "🛡️ <b>Confidence Gate Standby (<65% Threshold):</b> Active classifier confidence is sufficient. If an unseen anomaly occurs, operators are routed to workshop overhaul."
    )
    st.markdown(f"""
    <div class="guidance-box" style="background-color:{unk_bg}; border-color:{unk_border};">
        <div class="card-title">Fallback Alert (Unknown Faults)</div>
        <div style="font-size:0.9rem; color:#FFD43B; line-height:1.55;">
            {unk_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

with b3:
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    x_vals = list(range(len(history_df)))
    if show_vib:
        fig_trend.add_trace(
            go.Scatter(x=x_vals, y=history_df["vibration_rms"], name="Vibration (g)", line=dict(color="#FCC419", width=2)),
            secondary_y=False
        )
    if show_pwr:
        fig_trend.add_trace(
            go.Scatter(x=x_vals, y=history_df["power_kw"] * 1000.0, name="Power (W)", line=dict(color="#4DABF7", width=2)),
            secondary_y=True
        )
    if show_flow:
        fig_trend.add_trace(
            go.Scatter(x=x_vals, y=history_df["water_flow_lpm"], name="Flow (L/min)", line=dict(color="#51CF66", width=1.5, dash="dot")),
            secondary_y=False
        )
    if show_press:
        fig_trend.add_trace(
            go.Scatter(x=x_vals, y=history_df["outlet_pressure_bar"], name="Pressure (Bar)", line=dict(color="#FF8787", width=1.5)),
            secondary_y=False
        )

    fig_trend.update_layout(
        title={"text": "Vibration vs. Power Consumption & Utility", "font": {"size": 14, "color": "#CED4DA"}},
        paper_bgcolor="#1B2026",
        plot_bgcolor="#14181C",
        height=205,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#CED4DA")),
        xaxis=dict(showgrid=False, color="#868E96"),
        yaxis=dict(showgrid=True, gridcolor="#252A30", color="#FCC419"),
        yaxis2=dict(showgrid=False, color="#4DABF7")
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

# Bottom Persistent Status Bar matching the screenshot footer
st.markdown(f"""
<div class="unknown-banner">
    ⚠️ <b>UNKNOWN FAULT fallback rule active:</b> "If classifier confidence drops below 65%, alert operator: Fault unrecognized with the current dataset. Please route motor to workshop for manual overhaul."
    &nbsp;|&nbsp; <b>Total Plant Utility Efficiency:</b> {latest['specific_energy_whl']} Wh/L pumped.
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    time.sleep(3)
    st.rerun()
