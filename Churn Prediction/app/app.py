import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import preprocess_data
from logistic_regression import LogisticRegression
from utils import load_object

# Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for UI styling (Updated for Dark Mode)
st.markdown("""
<style>
    /* Global Overrides */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Force all custom card text to be visible */ 
    .stMarkdown p { color: #c9d8e8 !important; } 
    div[data-testid="stMarkdownContainer"] p { color: #c9d8e8 !important; } 
    
    /* Section headers */ 
    h1, h2, h3 { color: #ffffff !important; } 
    
    /* Metric values */ 
    div[data-testid="metric-container"] label { color: #a8c4e0 !important; } 
    div[data-testid="metric-container"] div { color: #ffffff !important; } 

    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .customer-card {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e2a3a, #162032);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2e4a6a;
    }
    .risk-safe { border-left: 10px solid #22c55e !important; }
    .risk-watch { border-left: 10px solid #eab308 !important; }
    .risk-high { border-left: 10px solid #f97316 !important; }
    .risk-critical { border-left: 10px solid #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

# Helper for rendering dark-mode cards
def render_intervention_card(icon, title, description, button_label):
    card_html = f""" 
    <div style=" 
        background: linear-gradient(135deg, #1e2a3a, #162032); 
        border: 1px solid #2e4a6a; 
        border-left: 4px solid #3b9eff; 
        border-radius: 12px; 
        padding: 20px 24px; 
        margin-bottom: 16px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); 
    "> 
        <div style="font-size: 28px; margin-bottom: 10px;">{icon}</div> 
        <div style=" 
            color: #ffffff; 
            font-size: 16px; 
            font-weight: 700; 
            margin-bottom: 8px; 
            letter-spacing: 0.3px; 
        ">{title}</div> 
        <div style=" 
            color: #a8c4e0; 
            font-size: 14px; 
            line-height: 1.6; 
            margin-bottom: 16px; 
        ">{description}</div> 
    </div> 
    """ 
    st.markdown(card_html, unsafe_allow_html=True) 
    st.button(button_label, key=f"btn_{title[:10]}_{np.random.randint(0,10000)}", type="primary")

# Load Model and Preprocessor
@st.cache_resource
def load_models():
    model = load_object('models/custom_lr_model.pkl')
    preprocessor = load_object('models/preprocessor.pkl')
    threshold_data = load_object('models/threshold.pkl')
    return model, preprocessor, threshold_data['threshold']

try:
    model, preprocessor, threshold = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Helper: Risk Tier Classification
def get_risk_tier(prob):
    prob_pct = prob * 100
    if prob_pct <= 30:
        return 1, "SAFE", "🟢", "#22c55e", "risk-safe"
    elif prob_pct <= 50:
        return 2, "WATCH", "🟡", "#eab308", "risk-watch"
    elif prob_pct <= 75:
        return 3, "HIGH RISK", "🟠", "#f97316", "risk-high"
    else:
        return 4, "CRITICAL", "🔴", "#ef4444", "risk-critical"

# Helper: Generate Predictions
def predict_risk(data_dict):
    df = pd.DataFrame([data_dict])
    X = preprocess_data(df, is_training=False, preprocessor=preprocessor)
    prob = model.predict_proba(X.values)[0]
    return prob, X

# --- SIDEBAR: Session History ---
with st.sidebar:
    st.header("📋 Session History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df[['#', 'Tenure', 'Monthly', 'Churn %', 'Tier']])
        
        # Export as CSV
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )
    else:
        st.info("No predictions yet.")

# --- MAIN UI ---
st.title("Customer Churn Predictor")
st.subheader("Subscription Platform Risk Analyzer")

# Input Form
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.radio("Senior Citizen", ["Yes", "No"])
        partner = st.radio("Partner", ["Yes", "No"])
        dependents = st.radio("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.radio("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        monthly_charges = st.number_input("Monthly Charges", 18.0, 120.0, 65.0)
        total_charges_default = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, float(total_charges_default))
    submit = st.form_submit_button("Predict Churn Risk")

if submit:
    input_data = {
        'gender': gender, 'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'Contract': contract,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    
    prob, X_processed = predict_risk(input_data)
    tier_num, tier_label, tier_icon, tier_color, tier_class = get_risk_tier(prob)
    
    # Add to History
    st.session_state.history.append({
        '#': len(st.session_state.history) + 1,
        'Tenure': tenure,
        'Monthly': f"${monthly_charges:.2f}",
        'Churn %': f"{prob*100:.1f}%",
        'Tier': tier_label
    })

    st.markdown("---")
    
    tab1, tab2 = st.tabs(["💙 Customer View", "📊 Agent Action Center"])
    
    with tab1:
        # TIER CARD
        st.markdown(f"""
        <div class="customer-card {tier_class}">
            <h2 style="color:{tier_color}; margin:0;">{tier_icon} {tier_label}</h2>
            <p style="font-size:1.2em; margin-top:10px; color:#ffffff !important;">Your current churn probability is <b>{prob*100:.1f}%</b>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PERSONALIZED INSTRUCTIONS
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">What You Can Do To Stay With Us 💙</h3>', unsafe_allow_html=True)
        instructions = []
        if contract == "Month-to-month" and prob >= 0.5:
            instructions.append(("📅", "Upgrade Plan", "Switching to an Annual or Two-Year plan saves you up to 20% and locks in your current rate.", "Claim Discount"))
        if tenure < 12 and prob >= 0.4:
            instructions.append(("🌱", "Early Access", "You're still in your early months! Customers who stay past 12 months save significantly on average billing.", "Learn More"))
        if online_security == "No" and internet_service != "No":
            instructions.append(("🔒", "Stay Secure", "Adding Online Security reduces account vulnerabilities and comes with a 30-day free trial.", "Activate Now"))
        if tech_support == "No" and monthly_charges > 60:
            instructions.append(("🛠️", "Free Support", "Premium Tech Support is included free for customers on plans above $60/month. Activate it today.", "Activate Today"))
        if multiple_lines == "No" and phone_service == "Yes":
            instructions.append(("📞", "Family Line", "Add a second line for a family member at 50% off your current line rate.", "Add Line"))
        if senior_citizen == "Yes":
            instructions.append(("🎖️", "Senior Loyalty", "You qualify for our Senior Loyalty Program — 15% monthly discount + priority support.", "Claim Discount"))
        if partner == "Yes" or dependents == "Yes":
            instructions.append(("👨‍👩‍👧‍👦", "Family Bundle", "Family Bundle plans offer shared data, multi-device protection, and unified billing.", "Explore Bundle"))
        if total_charges > 2000 and prob >= 0.5:
            instructions.append(("💰", "Loyalty Reward", "As a long-term valued customer, you are eligible for a Loyalty Reward Credit on your next bill.", "Claim Credit"))
        if internet_service == "Fiber optic" and online_backup == "No":
            instructions.append(("☁️", "Cloud Backup", "Fiber customers get free cloud backup (50GB) — enable it in your account settings now.", "Enable Now"))
        
        # Display max 4
        for icon, title, desc, btn in instructions[:4]:
            render_intervention_card(icon, title, desc, btn)

    with tab2:
        # REVENUE AT RISK
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">Predicted Revenue at Risk</h3>', unsafe_allow_html=True)
        monthly_risk = monthly_charges * prob
        annual_risk = monthly_charges * 12 * prob
        rev_color = "#22c55e" if monthly_risk < 20 else "#f97316" if monthly_risk <= 60 else "#ef4444"
        
        c1, c2 = st.columns(2)
        c1.metric("Monthly Revenue at Risk", f"${monthly_risk:.2f}")
        c2.metric("Annual Revenue at Risk", f"${annual_risk:.2f}")
        st.markdown(f"<p style='color:{rev_color}; font-weight:bold;'>Estimated Risk Impact Level</p>", unsafe_allow_html=True)
        
        # IMMEDIATE ACTIONS
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">Section A: IMMEDIATE ACTIONS</h3>', unsafe_allow_html=True)
        if tier_num == 4: st.error("🚨 Trigger retention call from senior account manager")
        elif tier_num == 3: st.warning("✉️ Send personalized email with exclusive offer within 24hrs")
        elif tier_num == 2: st.info("📧 Add to weekly automated re-engagement email sequence")
        else: st.success("✅ No immediate action needed — continue standard engagement")
        
        # RECOMMENDED OFFERS
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">Section B: RECOMMENDED OFFER</h3>', unsafe_allow_html=True)
        offers = []
        if monthly_charges > 80: offers.append(("💸", "Price Optimization", "Offer 10% bill reduction for 3 months", "Apply Offer"))
        if contract == "Month-to-month": offers.append(("📅", "Contract Upgrade", "Offer contract upgrade incentive: 2 free months if they switch to annual", "Propose Upgrade"))
        if tenure < 6: offers.append(("🤝", "Onboarding Support", "Offer onboarding support call + first month credit", "Schedule Call"))
        if senior_citizen == "Yes": offers.append(("🎖️", "Senior Program", "Enroll in senior loyalty program proactively", "Enroll Now"))
        if tech_support == "No" and device_protection == "No": offers.append(("📦", "Bundle Offer", "Bundle upsell: Security + Protection pack at 40% off", "Suggest Bundle"))
        
        for icon, title, desc, btn in offers:
            render_intervention_card(icon, title, desc, btn)
        
        # CHURN DRIVER ANALYSIS
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">Section C: CHURN DRIVER ANALYSIS</h3>', unsafe_allow_html=True)
        weights = model.weights
        contributions = weights * X_processed.values[0]
        feat_names = preprocessor['columns']
        contrib_df = pd.DataFrame({'Feature': feat_names, 'Impact': contributions}).sort_values(by='Impact', ascending=False).head(3)
        st.bar_chart(contrib_df.set_index('Feature'))

    # WHAT-IF SIMULATOR
    if prob >= 0.3:
        st.markdown("---")
        st.markdown('<h3 style="color: #ffffff; font-weight: 700;">🔄 What-If Simulator — How to Reduce This Customer\'s Risk</h3>', unsafe_allow_html=True)
        
        sims = []
        # Sim 1: Upgrade Contract
        sim1_data = input_data.copy(); sim1_data['Contract'] = "Two year"
        p1, _ = predict_risk(sim1_data); sims.append(("Upgrade to Two-Year Contract", p1))
        
        # Sim 2: Add Security + TechSupport
        sim2_data = input_data.copy(); sim2_data['OnlineSecurity'] = "Yes"; sim2_data['TechSupport'] = "Yes"
        p2, _ = predict_risk(sim2_data); sims.append(("Add Security + TechSupport", p2))
        
        # Sim 3: Stay 12 more months
        sim3_data = input_data.copy(); sim3_data['tenure'] += 12
        p3, _ = predict_risk(sim3_data); sims.append(("Stay 12 more months", p3))
        
        cols = st.columns(3)
        for i, (label, new_p) in enumerate(sims):
            delta = (prob - new_p) * 100
            color = "#22c55e" if delta > 10 else "#eab308" if delta > 5 else "#c9d8e8"
            cols[i].markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1e2a3a, #162032); 
                border: 1px solid #2e4a6a; 
                border-radius: 12px; 
                padding: 20px; 
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
                <div style="color: #ffffff; font-weight: 700; margin-bottom: 10px;">{label}</div>
                <div style="font-size: 1.5em; color: #ffffff; margin: 10px 0;">{prob*100:.1f}% → {new_p*100:.1f}%</div>
                <div style="color: {color}; font-weight: bold;">↓ reduced by {delta:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
