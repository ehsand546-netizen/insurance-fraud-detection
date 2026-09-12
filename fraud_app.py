import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Insurance Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    padding: 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.result-fraud {
    background: #fff0f0;
    border: 2px solid #e94560;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-legit {
    background: #f0fff4;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.metric-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Claim Fraud Detection System</h1>
    <p>Powered by Machine Learning (Random Forest) | Minor Project — B.Tech CSE Data Science</p>
    <p>MD Ehsan Khan | Ayush Jalal | Sampurna Sinha</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    df = pd.read_csv('data/insurance_claims (1).csv')
    df = df.drop(columns=['_c39'])
    mode_val = df['authorities_contacted'].mode()[0]
    df['authorities_contacted'] = df['authorities_contacted'].fillna(mode_val)
    df = df.drop_duplicates()
    cols_drop = ['policy_number','policy_bind_date','incident_date','insured_zip','incident_location']
    df = df.drop(columns=cols_drop)
    df['fraud_reported'] = df['fraud_reported'].map({'Y':1,'N':0})
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object','str']).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    X = df.drop(columns=['fraud_reported'])
    y = df['fraud_reported']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, scaler, X.columns.tolist(), df

model, scaler, feature_names, df_ref = train_model()

st.markdown("### 📋 Enter Claim Details")
st.markdown("Fill in the details below and click **Predict** to check if this claim is fraudulent.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Policyholder Info**")
    age = st.slider("Age of Policyholder", 18, 70, 35)
    months_as_customer = st.slider("Months as Customer", 1, 500, 100)
    insured_sex = st.selectbox("Gender", ["MALE", "FEMALE"])
    insured_education_level = st.selectbox("Education Level", ["High School", "College", "Associate", "Bachelor", "Masters", "MD", "JD"])
    insured_relationship = st.selectbox("Relationship", ["husband", "own-child", "unmarried", "wife", "not-in-family", "other-relative"])

with col2:
    st.markdown("**📋 Policy Info**")
    policy_state = st.selectbox("Policy State", ["OH", "IN", "IL", "ID", "VA", "WV", "NC", "SC", "NY", "PA"])
    policy_csl = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
    policy_deductable = st.selectbox("Deductible ($)", [500, 1000, 2000])
    policy_annual_premium = st.number_input("Annual Premium ($)", 500.0, 3000.0, 1200.0, step=50.0)
    umbrella_limit = st.selectbox("Umbrella Limit ($)", [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000])

with col3:
    st.markdown("**🚗 Incident Info**")
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"])
    incident_severity = st.selectbox("Incident Severity ⚠️", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
    collision_type = st.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision", "?"])
    number_of_vehicles = st.slider("Number of Vehicles Involved", 1, 4, 1)
    bodily_injuries = st.slider("Bodily Injuries", 0, 2, 0)
    witnesses = st.slider("Number of Witnesses", 0, 3, 1)

col4, col5 = st.columns(2)
with col4:
    st.markdown("**💰 Claim Amounts**")
    total_claim_amount = st.number_input("Total Claim Amount ($)", 100, 120000, 50000, step=1000)
    injury_claim = st.number_input("Injury Claim ($)", 0, 30000, 5000, step=500)
    property_claim = st.number_input("Property Claim ($)", 0, 30000, 5000, step=500)
    vehicle_claim = st.number_input("Vehicle Claim ($)", 0, 80000, 40000, step=1000)

with col5:
    st.markdown("**🔍 Additional Info**")
    property_damage = st.selectbox("Property Damage?", ["YES", "NO", "?"])
    police_report = st.selectbox("Police Report Available?", ["YES", "NO", "?"])
    authorities = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"])
    auto_year = st.slider("Vehicle Year", 1995, 2015, 2005)
    incident_hour = st.slider("Incident Hour (0-23)", 0, 23, 12)
    capital_gains = st.number_input("Capital Gains ($)", 0, 100000, 0, step=1000)
    capital_loss = st.number_input("Capital Loss ($)", 0, 100000, 0, step=1000)

st.markdown("---")
predict_btn = st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True)

if predict_btn:
    encodings = {
        'insured_sex': {'MALE': 1, 'FEMALE': 0},
        'policy_state': {'OH': 4, 'IN': 2, 'IL': 1, 'ID': 0, 'VA': 7, 'WV': 8, 'NC': 3, 'SC': 6, 'NY': 5, 'PA': 5},
        'policy_csl': {'100/300': 0, '250/500': 1, '500/1000': 2},
        'incident_type': {'Single Vehicle Collision': 3, 'Multi-vehicle Collision': 1, 'Vehicle Theft': 4, 'Parked Car': 2},
        'incident_severity': {'Trivial Damage': 3, 'Minor Damage': 2, 'Major Damage': 0, 'Total Loss': 1},
        'collision_type': {'Front Collision': 0, 'Rear Collision': 2, 'Side Collision': 3, '?': 1},
        'property_damage': {'YES': 2, 'NO': 1, '?': 0},
        'police_report': {'YES': 2, 'NO': 1, '?': 0},
        'authorities': {'Police': 3, 'Fire': 1, 'Ambulance': 0, 'Other': 2, 'None': 2},
        'insured_education_level': {'High School': 3, 'College': 1, 'Associate': 0, 'Bachelor': 1, 'Masters': 4, 'MD': 5, 'JD': 2},
        'insured_relationship': {'husband': 1, 'own-child': 3, 'unmarried': 5, 'wife': 6, 'not-in-family': 2, 'other-relative': 4},
    }

    input_data = {
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_state': encodings['policy_state'].get(policy_state, 0),
        'policy_csl': encodings['policy_csl'].get(policy_csl, 0),
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': encodings['insured_sex'].get(insured_sex, 0),
        'insured_education_level': encodings['insured_education_level'].get(insured_education_level, 0),
        'insured_occupation': 3,
        'insured_hobbies': 5,
        'insured_relationship': encodings['insured_relationship'].get(insured_relationship, 0),
        'capital-gains': capital_gains,
        'capital-loss': capital_loss,
        'incident_type': encodings['incident_type'].get(incident_type, 0),
        'collision_type': encodings['collision_type'].get(collision_type, 0),
        'incident_severity': encodings['incident_severity'].get(incident_severity, 0),
        'authorities_contacted': encodings['authorities'].get(authorities, 0),
        'incident_state': 4,
        'incident_city': 5,
        'incident_hour_of_the_day': incident_hour,
        'number_of_vehicles_involved': number_of_vehicles,
        'property_damage': encodings['property_damage'].get(property_damage, 0),
        'bodily_injuries': bodily_injuries,
        'witnesses': witnesses,
        'police_report_available': encodings['police_report'].get(police_report, 0),
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'auto_make': 5,
        'auto_model': 10,
        'auto_year': auto_year,
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    fraud_prob = round(probability[1] * 100, 1)
    legit_prob = round(probability[0] * 100, 1)

    st.markdown("### 🎯 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="result-fraud">
            <h2>🚨 HIGH FRAUD RISK DETECTED</h2>
            <h3 style="color: #e94560;">Fraud Probability: {fraud_prob}%</h3>
            <p>This claim shows characteristics consistent with fraudulent activity.<br>
            <strong>Recommendation: Refer to human investigator for detailed review.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-legit">
            <h2>✅ CLAIM APPEARS LEGITIMATE</h2>
            <h3 style="color: #22c55e;">Legitimate Probability: {legit_prob}%</h3>
            <p>This claim does not show strong indicators of fraud.<br>
            <strong>Recommendation: Proceed with standard claim processing.</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Probability Breakdown")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Fraud Probability", f"{fraud_prob}%",
                  delta="HIGH RISK" if fraud_prob > 50 else "LOW RISK")
    with col_r2:
        st.metric("Legitimate Probability", f"{legit_prob}%")
    with col_r3:
        st.metric("Model Used", "Random Forest", delta="88.08% Accuracy")

    st.markdown("### ⚠️ Key Risk Factors")
    risk_factors = []
    if incident_severity in ["Major Damage", "Total Loss"]:
        risk_factors.append("🔴 High incident severity — strongest fraud indicator")
    if total_claim_amount > 80000:
        risk_factors.append("🔴 Very high claim amount")
    if vehicle_claim > 50000:
        risk_factors.append("🟡 High vehicle claim amount")
    if bodily_injuries > 1:
        risk_factors.append("🟡 Multiple bodily injuries reported")
    if witnesses == 0:
        risk_factors.append("🟡 No witnesses present")
    if property_damage == "?":
        risk_factors.append("🟡 Property damage status unknown")
    if police_report == "NO":
        risk_factors.append("🟡 No police report filed")

    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("✅ No major risk factors detected")

st.markdown("---")
st.markdown("### 📈 Model Performance Summary")
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
metrics = [
    ("Accuracy", "88.08%"), ("Precision", "89.24%"),
    ("Recall", "88.12%"), ("F1-Score", "88.68%"), ("ROC-AUC", "88.08%")
]
for col, (name, val) in zip([col_m1, col_m2, col_m3, col_m4, col_m5], metrics):
    with col:
        st.metric(name, val)

st.markdown("""
---
<center><small>
Insurance Claim Fraud Detection | Random Forest Model | 
B.Tech CSE (Data Science) Minor Project 2026-27 | 
Amity School of Engineering & Technology
</small></center>
""", unsafe_allow_html=True)
