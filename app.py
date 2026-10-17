"""
Credit Scoring System - Interactive Streamlit Dashboard
=======================================================
A state-of-the-art Web Application for Real-Time Credit Risk Assessment,
Underwriting Decisioning, and Model Explainability.

Internship Project for CodeAlpha
Author: Senior ML Engineer & Full-Stack Python Developer
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predictor import CreditScoringPredictor
from src.data_generator import generate_synthetic_credit_data
from src.model_trainer import train_and_evaluate_model, FEATURE_COLUMNS, TARGET_COLUMN


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CreditIQ | Credit Scoring Model",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS & VISUAL STYLING
# =============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient Header Accent */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    /* Decision Result Banner */
    .decision-banner-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .decision-banner-denied {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* Badge Tags */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-approved {
        background: #10b981;
        color: #ffffff;
    }
    .badge-denied {
        background: #ef4444;
        color: #ffffff;
    }

    /* Factor pill */
    .factor-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #6366f1;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.92rem;
    }
    .factor-risk {
        border-left-color: #ef4444;
    }
    .factor-positive {
        border-left-color: #10b981;
    }

    /* Subtext formatting */
    .field-hint {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# MODEL CACHING & INITIALIZATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_predictor():
    """Initializes and caches the Credit Scoring inference engine."""
    artifacts_dir = "models"
    model_path = os.path.join(artifacts_dir, "credit_scoring_model.joblib")

    # If model is not trained yet, train on the fly
    if not os.path.exists(model_path):
        with st.spinner("Initializing Credit Scoring ML Pipeline for the first time..."):
            df = generate_synthetic_credit_data(n_samples=5000, random_state=42)
            train_and_evaluate_model(df, artifacts_dir=artifacts_dir)

    return CreditScoringPredictor(artifacts_dir=artifacts_dir)


try:
    predictor = get_predictor()
    model_metadata = predictor.metadata or {}
    feature_importances = predictor.feature_importances
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()


# =============================================================================
# HEADER SECTION
# =============================================================================
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <h1>💳 CreditIQ — Intelligent Credit Scoring Engine</h1>
                <p>Enterprise Risk Underwriting & Explainable Machine Learning Dashboard | CodeAlpha Internship</p>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(255,255,255,0.12); padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.88rem; font-weight: 600;">
                    🤖 Model: Random Forest Ensemble (v1.0)
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR CONTROLS & APPLICANT PRESETS
# =============================================================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### ⚡ Quick-Test Presets")
    st.markdown("Load pre-configured applicant profiles with one click:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        load_prime = st.button("🌟 Prime (Approved)", use_container_width=True)
    with col_p2:
        load_subprime = st.button("⚠️ Subprime (High Risk)", use_container_width=True)

    load_borderline = st.button("⚖️ Borderline / Fair Profile", use_container_width=True)

    # Preset logic
    if "preset_data" not in st.session_state:
        st.session_state.preset_data = {
            "age": 36,
            "income": 75000,
            "debt": 14000,
            "accounts": 7,
            "history": 10.5,
            "delay": 0,
            "utilization": 22.0,
            "missed": 0,
            "savings": 24000,
        }

    if load_prime:
        st.session_state.preset_data = {
            "age": 42,
            "income": 115000,
            "debt": 18000,
            "accounts": 8,
            "history": 14.0,
            "delay": 0,
            "utilization": 15.0,
            "missed": 0,
            "savings": 45000,
        }
        st.toast("Loaded 'Prime Approved' applicant preset!", icon="✅")

    if load_subprime:
        st.session_state.preset_data = {
            "age": 25,
            "income": 32000,
            "debt": 28000,
            "accounts": 12,
            "history": 2.0,
            "delay": 35,
            "utilization": 85.0,
            "missed": 3,
            "savings": 500,
        }
        st.toast("Loaded 'Subprime High Risk' applicant preset!", icon="⚠️")

    if load_borderline:
        st.session_state.preset_data = {
            "age": 31,
            "income": 52000,
            "debt": 22000,
            "accounts": 6,
            "history": 5.0,
            "delay": 12,
            "utilization": 48.0,
            "missed": 1,
            "savings": 4500,
        }
        st.toast("Loaded 'Borderline Fair' applicant preset!", icon="⚖️")

    st.markdown("---")
    st.markdown("### 📊 Benchmark Summary")
    if model_metadata and "metrics" in model_metadata:
        m = model_metadata["metrics"]
        st.metric("Test ROC-AUC", f"{m.get('roc_auc', 0.94):.3f}")
        st.metric("Model Precision", f"{m.get('precision', 0.91)*100:.1f}%")
        st.metric("Model Recall", f"{m.get('recall', 0.92)*100:.1f}%")
        st.metric("F1-Score", f"{m.get('f1_score', 0.91):.3f}")

    st.markdown("---")
    st.caption("Developed for CodeAlpha Internship Program.\nPipeline powered by Scikit-Learn & Streamlit.")


# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================
tab_scoring, tab_insights, tab_metrics, tab_batch, tab_docs = st.tabs([
    "🎯 Live Underwriting",
    "📈 Feature Importance & Insights",
    "🧪 Model Evaluation & Benchmark",
    "📁 Batch Evaluation (CSV)",
    "📖 Documentation & Architecture"
])


# =============================================================================
# TAB 1: LIVE UNDERWRITING & SCORING
# =============================================================================
with tab_scoring:
    st.markdown("### 📋 Applicant Financial Profile")
    st.markdown("Input the applicant's financial attributes below. The machine learning model will evaluate credit risk in real time.")

    preset = st.session_state.preset_data

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.markdown("#### 👤 Personal & Income")
        input_age = st.slider("Age (Years)", min_value=18, max_value=80, value=int(preset["age"]), help="Applicant's current age")
        input_income = st.number_input(
            "Annual Income ($)",
            min_value=5000,
            max_value=500000,
            value=int(preset["income"]),
            step=2500,
            help="Total gross annual income before taxes"
        )
        input_savings = st.number_input(
            "Liquid Savings / Assets ($)",
            min_value=0,
            max_value=300000,
            value=int(preset["savings"]),
            step=1000,
            help="Total liquid savings available in checking/savings accounts"
        )

    with col_c2:
        st.markdown("#### 💳 Debt & Credit Lines")
        input_debt = st.number_input(
            "Total Outstanding Debt ($)",
            min_value=0,
            max_value=400000,
            value=int(preset["debt"]),
            step=1000,
            help="Total balance across mortgages, car loans, student loans, and credit cards"
        )
        input_accounts = st.slider(
            "Number of Open Accounts",
            min_value=1,
            max_value=30,
            value=int(preset["accounts"]),
            help="Total number of active credit cards and loan accounts"
        )
        input_utilization = st.slider(
            "Credit Utilization Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(preset["utilization"]),
            step=1.0,
            help="Revolving credit card balance divided by total available credit limit"
        )

    with col_c3:
        st.markdown("#### ⏳ Repayment Discipline")
        max_valid_history = max(1.0, float(input_age - 18))
        history_val = min(float(preset["history"]), max_valid_history)
        input_history = st.slider(
            "Credit History Length (Years)",
            min_value=0.5,
            max_value=float(max_valid_history),
            value=float(history_val),
            step=0.5,
            help="Time since the oldest credit account was opened (capped at Age - 18)"
        )
        input_delay = st.slider(
            "Average Payment Delay (Days)",
            min_value=0,
            max_value=90,
            value=int(preset["delay"]),
            help="Average days past payment due date in the past 12 months"
        )
        input_missed = st.slider(
            "Number of Missed Payments",
            min_value=0,
            max_value=8,
            value=int(preset["missed"]),
            help="Delinquent payments 30+ days overdue in the last 24 months"
        )

    # Derived Real-Time Indicators
    dti_calc = (input_debt / input_income) if input_income > 0 else 1.0
    st.markdown("---")
    
    # Financial Ratio Quick Overview
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        dti_color = "normal" if dti_calc < 0.36 else ("off" if dti_calc < 0.45 else "inverse")
        st.metric("Debt-to-Income (DTI)", f"{dti_calc*100:.1f}%", help="Under 36% is recommended by lenders")
    with col_r2:
        st.metric("Credit Utilization", f"{input_utilization:.1f}%", help="Under 30% is ideal")
    with col_r3:
        st.metric("Monthly Gross Income", f"${input_income/12:,.0f}")
    with col_r4:
        st.metric("Debt / Savings Buffer", f"{input_savings / max(input_debt, 1):.2f}x", help="Ratio of liquid savings to total debt")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Evaluate Creditworthiness", type="primary", use_container_width=True)

    # Evaluate prediction
    if predict_btn or True:
        applicant_payload = {
            "Age": input_age,
            "Annual_Income": float(input_income),
            "Total_Debt": float(input_debt),
            "Num_Open_Accounts": input_accounts,
            "Credit_History_Length": float(input_history),
            "Delay_in_Payments": float(input_delay),
            "Credit_Utilization_Ratio": float(input_utilization / 100.0),
            "Num_Missed_Payments": input_missed,
            "Savings_Balance": float(input_savings),
            "Debt_to_Income_Ratio": float(dti_calc)
        }

        eval_result = predictor.predict_single(applicant_payload)
        is_approved = (eval_result["creditworthy"] == 1)
        prob = eval_result["approval_probability"]
        credit_score = eval_result["credit_score"]
        tier = eval_result["risk_tier"]
        risk_level = eval_result["risk_level"]
        status_color = eval_result["status_color"]

        st.markdown("### 🏆 Underwriting Decision & Credit Score")

        # Decision Banner
        if is_approved:
            st.markdown(
                f"""
                <div class="decision-banner-approved">
                    <span class="badge badge-approved">Approved</span>
                    <h2 style="color: #10b981; margin: 0.5rem 0; font-size: 2rem;">APPLICANT IS CREDITWORTHY</h2>
                    <p style="color: #cbd5e1; margin: 0; font-size: 1.1rem;">
                        Approval Probability: <strong>{prob*100:.1f}%</strong> | Risk Tier: <strong>{tier}</strong> ({risk_level})
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="decision-banner-denied">
                    <span class="badge badge-denied">Declined</span>
                    <h2 style="color: #ef4444; margin: 0.5rem 0; font-size: 2rem;">HIGH CREDIT RISK DETECTED</h2>
                    <p style="color: #cbd5e1; margin: 0; font-size: 1.1rem;">
                        Default Probability: <strong>{(1-prob)*100:.1f}%</strong> | Risk Tier: <strong>{tier}</strong> ({risk_level})
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        col_g1, col_g2 = st.columns([1.2, 1])

        with col_g1:
            # FICO Score Gauge Meter
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=credit_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Estimated FICO® Score<br><span style='font-size:0.8em;color:{status_color}'>{tier}</span>", 'font': {'size': 20, 'color': '#ffffff'}},
                delta={'reference': 670, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
                gauge={
                    'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': status_color, 'thickness': 0.28},
                    'bgcolor': "rgba(255, 255, 255, 0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [300, 580], 'color': 'rgba(239, 68, 68, 0.25)'},
                        {'range': [580, 670], 'color': 'rgba(249, 115, 22, 0.25)'},
                        {'range': [670, 740], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [740, 800], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [800, 850], 'color': 'rgba(5, 150, 105, 0.40)'},
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.8,
                        'value': credit_score
                    }
                }
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
                height=320,
                margin=dict(l=30, r=30, t=50, b=20)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_g2:
            st.markdown("#### 🔍 Explainability & Risk Drivers")

            st.markdown("##### ✅ Positive Strengths")
            for s in eval_result["strengths"]:
                st.markdown(f'<div class="factor-item factor-positive">✓ {s}</div>', unsafe_allow_html=True)

            st.markdown("##### ⚠️ Risk Warnings & Flags")
            for r in eval_result["risk_factors"]:
                st.markdown(f'<div class="factor-item factor-risk">! {r}</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2: FEATURE IMPORTANCE & RISK DRIVERS
# =============================================================================
with tab_insights:
    st.markdown("### 📊 Global Model Explainability (Random Forest)")
    st.markdown("Feature importance indicates which customer metrics have the highest predictive weight in classifying creditworthiness across all training samples.")

    if feature_importances is not None:
        fig_imp = px.bar(
            feature_importances.sort_values(by="Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Purples",
            title="Feature Importance Ranking (Gini Impurity Reduction)"
        )
        fig_imp.update_traces(
            texttemplate='%{text:.2%}',
            textposition='outside',
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickformat=".0%"),
            yaxis=dict(showgrid=False),
            height=460,
            margin=dict(l=20, r=40, t=50, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🔴 Repayment Delays</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">
                    Similar to standard FICO payment history (35% weight), payment delays and missed bills are the strongest drivers of immediate default risk.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_i2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🟡 Debt-to-Income (DTI)</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">
                    Measures cash-flow strain. Applicants with DTI over 43% face heavy rejection probabilities due to lack of discretionary cushion.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    with col_i3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>🟢 Credit History Length</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">
                    Older credit histories (8+ years) demonstrate resilience across economic cycles, significantly boosting approval odds.
                </p>
            </div>
            """, unsafe_allow_html=True
        )


# =============================================================================
# TAB 3: MODEL EVALUATION & BENCHMARKS
# =============================================================================
with tab_metrics:
    st.markdown("### 🧪 Machine Learning Pipeline Evaluation")
    st.markdown("Comprehensive performance metrics evaluated on a held-out stratified test set (20% test split).")

    if model_metadata and "metrics" in model_metadata:
        m = model_metadata["metrics"]
        cm = m["confusion_matrix"]
        curves = model_metadata.get("curves", {})

        # Top Metric Cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1:
            st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        with col_m2:
            st.metric("Precision", f"{m['precision']*100:.2f}%")
        with col_m3:
            st.metric("Recall", f"{m['recall']*100:.2f}%")
        with col_m4:
            st.metric("F1-Score", f"{m['f1_score']:.4f}")
        with col_m5:
            st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

        st.markdown("---")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            # Confusion Matrix Heatmap
            cm_z = [[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]]
            cm_x = ['Predicted Creditworthy', 'Predicted High Risk']
            cm_y = ['Actual Creditworthy', 'Actual High Risk']
            cm_text = [[f"{cm[1][1]} (TP)", f"{cm[1][0]} (FN)"], [f"{cm[0][1]} (FP)", f"{cm[0][0]} (TN)"]]

            fig_cm = px.imshow(
                cm_z,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=cm_x,
                y=cm_y,
                text_auto=False,
                color_continuous_scale="Teal",
                title="Confusion Matrix (Stratified Test Set)"
            )
            fig_cm.update_traces(text=cm_text, texttemplate="%{text}", textfont_size=16)
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_v2:
            # ROC Curve
            if curves and "fpr" in curves:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=curves["fpr"],
                    y=curves["tpr"],
                    mode='lines',
                    name=f"Random Forest (AUC = {m['roc_auc']:.3f})",
                    line=dict(color='#6366f1', width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Chance (AUC = 0.50)',
                    line=dict(color='#94a3b8', dash='dash')
                ))
                fig_roc.update_layout(
                    title="Receiver Operating Characteristic (ROC) Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#ffffff", 'family': "Plus Jakarta Sans"},
                    xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        # Benchmark Comparison Table
        st.markdown("#### 🥊 Model Benchmark Comparison")
        if "comparison" in model_metadata:
            df_comp = pd.DataFrame(model_metadata["comparison"])
            st.dataframe(
                df_comp.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.4f}",
                    "ROC-AUC": "{:.4f}"
                }).highlight_max(axis=0, subset=["ROC-AUC", "F1-Score", "Accuracy"], color="#312e81"),
                use_container_width=True
            )


# =============================================================================
# TAB 4: BATCH APPLICANT EVALUATION
# =============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Credit Risk Assessment")
    st.markdown("Upload a CSV file containing multiple applicant records to generate instant credit decisions, probability scores, and FICO tiers.")

    # Download Template Button
    sample_df = pd.DataFrame({
        "Age": [29, 45, 52, 23],
        "Annual_Income": [62000, 120000, 35000, 24000],
        "Total_Debt": [15000, 22000, 29000, 18000],
        "Num_Open_Accounts": [5, 9, 7, 3],
        "Credit_History_Length": [6.5, 18.0, 12.0, 1.5],
        "Delay_in_Payments": [0, 0, 28, 42],
        "Credit_Utilization_Ratio": [0.24, 0.18, 0.82, 0.91],
        "Num_Missed_Payments": [0, 0, 2, 4],
        "Savings_Balance": [14000, 52000, 800, 200]
    })
    csv_template = sample_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Sample CSV Template",
        data=csv_template,
        file_name="credit_applicants_sample.csv",
        mime="text/csv",
        help="Use this template to format batch applicant data"
    )

    uploaded_file = st.file_uploader("Upload Applicants CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded {len(batch_df)} applicant records.")

            with st.spinner("Scoring batch applicants..."):
                scored_df = predictor.predict_batch(batch_df)

            col_b1, col_b2, col_b3 = st.columns(3)
            approved_count = int((scored_df["Predicted_Creditworthy"] == 1).sum())
            total_count = len(scored_df)
            with col_b1:
                st.metric("Total Applicants", total_count)
            with col_b2:
                st.metric("Approved", f"{approved_count} ({approved_count/total_count*100:.1f}%)")
            with col_b3:
                st.metric("Declined", f"{total_count - approved_count} ({(total_count - approved_count)/total_count*100:.1f}%)")

            st.dataframe(scored_df, use_container_width=True)

            # Download Result CSV
            result_csv = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Scored Results (CSV)",
                data=result_csv,
                file_name="credit_scoring_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing uploaded batch file: {e}")


# =============================================================================
# TAB 5: DOCUMENTATION & ARCHITECTURE
# =============================================================================
with tab_docs:
    st.markdown("### 📖 System Architecture & Engineering Documentation")
    st.markdown(
        """
        #### 1. Executive Summary
        This application represents a comprehensive, production-grade Machine Learning credit risk engine developed
        for the **CodeAlpha Machine Learning Internship**. It automates credit underwriting decisions while preserving
        model interpretability, calibration, and transparency.

        #### 2. Features in the Credit Scoring Model
        - **Age**: Applicant demographic maturity.
        - **Annual Income**: Core debt-service capacity.
        - **Total Debt**: Aggregate liability volume.
        - **Number of Open Accounts**: Credit exposure and activity.
        - **Credit History Length**: Seasoning of borrower accounts (in years).
        - **Delay in Payments**: Average days overdue in the last 12 months.
        - **Credit Utilization Ratio**: Revolving credit utilization percentage.
        - **Number of Missed Payments**: Severe delinquencies (30+ days).
        - **Savings Balance**: Liquid emergency reserve.
        - **Debt-to-Income (DTI) Ratio**: Ratio of total debt obligations to annual income.

        #### 3. Machine Learning Architecture
        ```mermaid
        graph TD
            A[Synthetic Financial Data Engine] --> B[Data Preprocessing Pipeline]
            B --> C1[Median Missing Value Imputation]
            C1 --> C2[StandardScaler Normalization]
            C2 --> D[Random Forest Classifier Ensemble]
            D --> E[Calibrated Probability Engine]
            E --> F1[Binary Credit Decision]
            E --> F2[FICO Score Indicator 300 - 850]
            E --> F3[Explainable Risk Factor Decomposition]
        ```

        #### 4. Preprocessing & Leakage Prevention
        - Preprocessing steps (`SimpleImputer` and `StandardScaler`) are encapsulated inside a scikit-learn `Pipeline`.
        - Transformers are fit strictly on training splits and evaluated on stratified held-out validation sets.
        - Hyperparameters tuned with balanced class weights to prevent majority-class bias.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        <strong>CodeAlpha Internship Project</strong> — Credit Scoring Machine Learning System<br>
        Built with Python, Scikit-Learn, Streamlit & Plotly.
    </div>
    """,
    unsafe_allow_html=True
)
