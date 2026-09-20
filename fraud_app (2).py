import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import datetime
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Insurance Fraud Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.result-fraud {
    background: #fff0f0;
    border: 2px solid #e94560;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.result-legit {
    background: #f0fff4;
    border: 2px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.risk-low    { background:#f0fff4; border:2px solid #22c55e; border-radius:8px; padding:8px; text-align:center; }
.risk-medium { background:#fffbeb; border:2px solid #f59e0b; border-radius:8px; padding:8px; text-align:center; }
.risk-high   { background:#fff0f0; border:2px solid #e94560; border-radius:8px; padding:8px; text-align:center; }
.metric-card { background:#f8f9fa; border-radius:8px; padding:1rem; text-align:center; border:1px solid #e0e0e0; }
.history-table { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_checked' not in st.session_state:
    st.session_state.total_checked = 0
if 'total_fraud' not in st.session_state:
    st.session_state.total_fraud = 0

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Insurance Claim Fraud Detection System <span style="font-size:14px;background:#e94560;padding:4px 10px;border-radius:20px;">PRO v2.0</span></h1>
    <p>Powered by 5 Machine Learning Models | Minor Project — B.Tech CSE Data Science</p>
    <p>MD Ehsan Khan | Ayush Jalal | Sampurna Sinha | Amity School of Engineering & Technology</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio("Select Page", [
        "🔍 Single Claim Prediction",
        "📁 Batch Prediction (CSV)",
        "📊 Dashboard & History",
        "ℹ️ About & Model Info"
    ])
    st.markdown("---")
    st.markdown("### 📈 Session Stats")
    st.metric("Claims Checked", st.session_state.total_checked)
    st.metric("Fraud Detected", st.session_state.total_fraud)
    if st.session_state.total_checked > 0:
        rate = round(st.session_state.total_fraud / st.session_state.total_checked * 100, 1)
        st.metric("Fraud Rate", f"{rate}%")
    st.markdown("---")
    st.markdown("### 🤖 Model Selector")
    selected_model = st.selectbox("Choose Algorithm", [
        "Random Forest (Recommended)",
        "XGBoost (Best Recall)",
        "SVM",
        "Decision Tree",
        "Logistic Regression",
        "All 5 Models (Compare)"
    ])
    st.markdown("---")
    st.markdown("🌐 [Live App](https://insurance-fraud-detection-ehsan.streamlit.app)")
    st.markdown("📂 [GitHub](https://github.com/ehsand546-netizen/insurance-fraud-detection)")

# ── TRAIN ALL MODELS ─────────────────────────────────────────
@st.cache_resource
def train_all_models():
    df = pd.read_csv('insurance_claims.csv')
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
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    }
    trained = {}
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
        trained[name] = model
        accuracies[name] = acc
    return trained, scaler, X.columns.tolist(), accuracies

with st.spinner("🤖 Loading and training all 5 models... Please wait..."):
    all_models, scaler, feature_names, accuracies = train_all_models()

# ── HELPER: GET RISK LEVEL ────────────────────────────────────
def get_risk_level(prob):
    if prob < 30:   return "LOW RISK", "🟢", "risk-low"
    elif prob < 60: return "MEDIUM RISK", "🟡", "risk-medium"
    elif prob < 80: return "HIGH RISK", "🔴", "risk-high"
    else:           return "VERY HIGH RISK", "🚨", "risk-high"

# ── HELPER: PREPARE INPUT ─────────────────────────────────────
def prepare_input(age, months_as_customer, insured_sex, insured_education_level,
                  insured_relationship, policy_state, policy_csl, policy_deductable,
                  policy_annual_premium, umbrella_limit, incident_type, incident_severity,
                  collision_type, number_of_vehicles, bodily_injuries, witnesses,
                  property_damage, police_report, authorities, auto_year, incident_hour,
                  capital_gains, capital_loss, total_claim_amount, injury_claim,
                  property_claim, vehicle_claim):
    encodings = {
        'insured_sex': {'MALE': 1, 'FEMALE': 0},
        'policy_state': {'OH':4,'IN':2,'IL':1,'ID':0,'VA':7,'WV':8,'NC':3,'SC':6,'NY':5,'PA':5},
        'policy_csl': {'100/300':0,'250/500':1,'500/1000':2},
        'incident_type': {'Single Vehicle Collision':3,'Multi-vehicle Collision':1,'Vehicle Theft':4,'Parked Car':2},
        'incident_severity': {'Trivial Damage':3,'Minor Damage':2,'Major Damage':0,'Total Loss':1},
        'collision_type': {'Front Collision':0,'Rear Collision':2,'Side Collision':3,'?':1},
        'property_damage': {'YES':2,'NO':1,'?':0},
        'police_report': {'YES':2,'NO':1,'?':0},
        'authorities': {'Police':3,'Fire':1,'Ambulance':0,'Other':2,'None':2},
        'insured_education_level': {'High School':3,'College':1,'Associate':0,'Bachelor':1,'Masters':4,'MD':5,'JD':2},
        'insured_relationship': {'husband':1,'own-child':3,'unmarried':5,'wife':6,'not-in-family':2,'other-relative':4},
    }
    input_data = {
        'months_as_customer': months_as_customer, 'age': age,
        'policy_state': encodings['policy_state'].get(policy_state,0),
        'policy_csl': encodings['policy_csl'].get(policy_csl,0),
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': encodings['insured_sex'].get(insured_sex,0),
        'insured_education_level': encodings['insured_education_level'].get(insured_education_level,0),
        'insured_occupation': 3, 'insured_hobbies': 5,
        'insured_relationship': encodings['insured_relationship'].get(insured_relationship,0),
        'capital-gains': capital_gains, 'capital-loss': capital_loss,
        'incident_type': encodings['incident_type'].get(incident_type,0),
        'collision_type': encodings['collision_type'].get(collision_type,0),
        'incident_severity': encodings['incident_severity'].get(incident_severity,0),
        'authorities_contacted': encodings['authorities'].get(authorities,0),
        'incident_state': 4, 'incident_city': 5,
        'incident_hour_of_the_day': incident_hour,
        'number_of_vehicles_involved': number_of_vehicles,
        'property_damage': encodings['property_damage'].get(property_damage,0),
        'bodily_injuries': bodily_injuries, 'witnesses': witnesses,
        'police_report_available': encodings['police_report'].get(police_report,0),
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim, 'property_claim': property_claim,
        'vehicle_claim': vehicle_claim, 'auto_make': 5, 'auto_model': 10, 'auto_year': auto_year,
    }
    return scaler.transform(pd.DataFrame([input_data]))

# ════════════════════════════════════════════════════════════
# PAGE 1: SINGLE CLAIM PREDICTION
# ════════════════════════════════════════════════════════════
if page == "🔍 Single Claim Prediction":

    # Claim ID section
    st.markdown("### 🔑 Claim Identification")
    id1, id2, id3 = st.columns(3)
    with id1: user_id = st.text_input("👤 User ID", placeholder="e.g. USR-2026-001")
    with id2: policy_number = st.text_input("📋 Policy Number", placeholder="e.g. POL-2026-123456")
    with id3: claim_id = st.text_input("🗂️ Claim ID", placeholder="e.g. CLM-2026-789")

    if user_id or policy_number:
        st.info(f"📌 Record: User **{user_id or 'N/A'}** | Policy **{policy_number or 'N/A'}** | Claim **{claim_id or 'N/A'}**")

    st.markdown("---")
    st.markdown("### 📋 Enter Claim Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Policyholder Info**")
        age = st.slider("Age", 18, 70, 35)
        months_as_customer = st.slider("Months as Customer", 1, 500, 100)
        insured_sex = st.selectbox("Gender", ["MALE","FEMALE"])
        insured_education_level = st.selectbox("Education", ["High School","College","Associate","Bachelor","Masters","MD","JD"])
        insured_relationship = st.selectbox("Relationship", ["husband","own-child","unmarried","wife","not-in-family","other-relative"])

    with col2:
        st.markdown("**📋 Policy Info**")
        policy_state = st.selectbox("State", ["OH","IN","IL","ID","VA","WV","NC","SC","NY","PA"])
        policy_csl = st.selectbox("Policy CSL", ["100/300","250/500","500/1000"])
        policy_deductable = st.selectbox("Deductible ($)", [500,1000,2000])
        policy_annual_premium = st.number_input("Annual Premium ($)", 500.0, 3000.0, 1200.0, step=50.0)
        umbrella_limit = st.selectbox("Umbrella Limit", [0,1000000,2000000,3000000,4000000,5000000,6000000,7000000])

    with col3:
        st.markdown("**🚗 Incident Info**")
        incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision","Multi-vehicle Collision","Vehicle Theft","Parked Car"])
        incident_severity = st.selectbox("Incident Severity ⚠️", ["Minor Damage","Major Damage","Total Loss","Trivial Damage"])
        collision_type = st.selectbox("Collision Type", ["Front Collision","Rear Collision","Side Collision","?"])
        number_of_vehicles = st.slider("Vehicles Involved", 1, 4, 1)
        bodily_injuries = st.slider("Bodily Injuries", 0, 2, 0)
        witnesses = st.slider("Witnesses", 0, 3, 1)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**💰 Claim Amounts**")
        total_claim_amount = st.number_input("Total Claim ($)", 100, 120000, 50000, step=1000)
        injury_claim = st.number_input("Injury Claim ($)", 0, 30000, 5000, step=500)
        property_claim = st.number_input("Property Claim ($)", 0, 30000, 5000, step=500)
        vehicle_claim = st.number_input("Vehicle Claim ($)", 0, 80000, 40000, step=1000)

    with col5:
        st.markdown("**🔍 Additional Info**")
        property_damage = st.selectbox("Property Damage?", ["YES","NO","?"])
        police_report = st.selectbox("Police Report?", ["YES","NO","?"])
        authorities = st.selectbox("Authorities Contacted", ["Police","Fire","Ambulance","Other","None"])
        auto_year = st.slider("Vehicle Year", 1995, 2015, 2005)
        incident_hour = st.slider("Incident Hour", 0, 23, 12)
        capital_gains = st.number_input("Capital Gains ($)", 0, 100000, 0, step=1000)
        capital_loss = st.number_input("Capital Loss ($)", 0, 100000, 0, step=1000)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True)

    if predict_btn:
        input_scaled = prepare_input(
            age, months_as_customer, insured_sex, insured_education_level,
            insured_relationship, policy_state, policy_csl, policy_deductable,
            policy_annual_premium, umbrella_limit, incident_type, incident_severity,
            collision_type, number_of_vehicles, bodily_injuries, witnesses,
            property_damage, police_report, authorities, auto_year, incident_hour,
            capital_gains, capital_loss, total_claim_amount, injury_claim,
            property_claim, vehicle_claim)

        st.markdown("### 🎯 Prediction Results")
        st.markdown(f"**📌 Claim:** {claim_id or 'N/A'} | **👤 User:** {user_id or 'N/A'} | **📋 Policy:** {policy_number or 'N/A'} | **📅** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}")

        # ── ALL 5 MODELS ──────────────────────────────────────
        if "All 5" in selected_model:
            st.markdown("#### 🤖 All 5 Model Predictions")
            model_cols = st.columns(5)
            all_probs = {}
            for i, (mname, model) in enumerate(all_models.items()):
                prob = round(model.predict_proba(input_scaled)[0][1]*100, 1)
                pred = model.predict(input_scaled)[0]
                all_probs[mname] = prob
                risk_label, risk_icon, _ = get_risk_level(prob)
                with model_cols[i]:
                    color = "#e94560" if pred==1 else "#22c55e"
                    st.markdown(f"""
                    <div style="border:2px solid {color};border-radius:8px;padding:10px;text-align:center;background:{'#fff0f0' if pred==1 else '#f0fff4'}">
                    <b style="font-size:11px">{mname}</b><br>
                    <span style="font-size:22px;font-weight:bold;color:{color}">{prob}%</span><br>
                    <small>{risk_icon} {risk_label}</small><br>
                    <small>Acc: {accuracies[mname]}%</small>
                    </div>""", unsafe_allow_html=True)
            avg_prob = round(sum(all_probs.values())/5, 1)
            st.markdown(f"#### 🏆 Consensus: Average Fraud Probability = **{avg_prob}%**")
            fraud_votes = sum(1 for p in all_probs.values() if p > 50)
            if fraud_votes >= 3:
                st.error(f"🚨 **{fraud_votes}/5 models say FRAUD** — HIGH FRAUD RISK DETECTED")
            else:
                st.success(f"✅ **{5-fraud_votes}/5 models say LEGITIMATE** — Claim appears legitimate")
            fraud_prob = avg_prob
            prediction = 1 if fraud_votes >= 3 else 0
        else:
            # Single model
            mname = selected_model.split(" (")[0]
            model = all_models[mname]
            prediction = model.predict(input_scaled)[0]
            prob_arr = model.predict_proba(input_scaled)[0]
            fraud_prob = round(prob_arr[1]*100, 1)
            legit_prob = round(prob_arr[0]*100, 1)
            risk_label, risk_icon, risk_class = get_risk_level(fraud_prob)

            if prediction == 1:
                st.markdown(f"""<div class="result-fraud">
                <h2>🚨 HIGH FRAUD RISK DETECTED</h2>
                <h3 style="color:#e94560">{risk_icon} {risk_label} — {fraud_prob}% Fraud Probability</h3>
                <p>Policy: <b>{policy_number or 'N/A'}</b> | User: <b>{user_id or 'N/A'}</b> | Model: <b>{mname}</b></p>
                <p><strong>Recommendation: Refer to human investigator for detailed review.</strong></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-legit">
                <h2>✅ CLAIM APPEARS LEGITIMATE</h2>
                <h3 style="color:#22c55e">{risk_icon} {risk_label} — {legit_prob}% Legitimate</h3>
                <p>Policy: <b>{policy_number or 'N/A'}</b> | User: <b>{user_id or 'N/A'}</b> | Model: <b>{mname}</b></p>
                <p><strong>Recommendation: Proceed with standard claim processing.</strong></p>
                </div>""", unsafe_allow_html=True)

        # ── RISK GAUGE ────────────────────────────────────────
        st.markdown("### 📊 Risk Assessment")
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1: st.metric("Fraud Probability", f"{fraud_prob}%")
        with mc2:
            risk_label, risk_icon, _ = get_risk_level(fraud_prob)
            st.metric("Risk Level", f"{risk_icon} {risk_label}")
        with mc3: st.metric("Model Used", mname if "All 5" not in selected_model else "5 Models")
        with mc4: st.metric("Prediction Time", datetime.datetime.now().strftime('%H:%M:%S'))

        # Risk bar
        bar_color = "#22c55e" if fraud_prob < 30 else "#f59e0b" if fraud_prob < 60 else "#e94560"
        st.markdown(f"""
        <div style="background:#e0e0e0;border-radius:10px;height:20px;margin:10px 0;">
        <div style="background:{bar_color};width:{fraud_prob}%;height:20px;border-radius:10px;
        display:flex;align-items:center;justify-content:center;color:white;font-size:11px;font-weight:bold;">
        {fraud_prob}%</div></div>
        <div style="display:flex;justify-content:space-between;font-size:11px;color:#666">
        <span>0% — Safe</span><span>30% — Suspicious</span><span>60% — High Risk</span><span>100% — Critical</span></div>
        """, unsafe_allow_html=True)

        # ── RISK FACTORS ──────────────────────────────────────
        st.markdown("### ⚠️ Risk Factors Identified")
        risk_factors = []
        if incident_severity in ["Major Damage","Total Loss"]:
            risk_factors.append(("🔴", "High incident severity", "Strongest fraud indicator (importance: 0.217)"))
        if total_claim_amount > 80000:
            risk_factors.append(("🔴", "Very high claim amount", f"${total_claim_amount:,} — significantly above average"))
        if vehicle_claim > 50000:
            risk_factors.append(("🔴", "High vehicle claim", f"${vehicle_claim:,} flagged"))
        if bodily_injuries > 1:
            risk_factors.append(("🟡", "Multiple bodily injuries", f"{bodily_injuries} injuries reported"))
        if witnesses == 0:
            risk_factors.append(("🟡", "No witnesses", "Unverifiable incident"))
        if property_damage == "?":
            risk_factors.append(("🟡", "Property damage unknown", "Missing information"))
        if police_report == "NO":
            risk_factors.append(("🟡", "No police report", "Reduces claim credibility"))
        if months_as_customer < 24:
            risk_factors.append(("🟡", "New customer", f"Only {months_as_customer} months — early claim pattern"))

        if risk_factors:
            for icon, factor, detail in risk_factors:
                st.markdown(f"{icon} **{factor}** — {detail}")
        else:
            st.success("✅ No major risk factors detected in this claim.")

        # ── DOWNLOAD REPORT ───────────────────────────────────
        st.markdown("### 📥 Download Report")
        report_data = {
            "Field": ["User ID","Policy Number","Claim ID","Date & Time","Age",
                      "Incident Severity","Total Claim Amount","Vehicle Claim",
                      "Bodily Injuries","Witnesses","Police Report",
                      "Fraud Probability","Risk Level","Prediction","Model Used","Recommendation"],
            "Value": [user_id or "N/A", policy_number or "N/A", claim_id or "N/A",
                      datetime.datetime.now().strftime('%d/%m/%Y %H:%M'),
                      age, incident_severity, f"${total_claim_amount:,}", f"${vehicle_claim:,}",
                      bodily_injuries, witnesses, police_report,
                      f"{fraud_prob}%", risk_label,
                      "FRAUD RISK" if prediction==1 else "LEGITIMATE",
                      mname if "All 5" not in selected_model else "5 Models (Consensus)",
                      "Refer to investigator" if prediction==1 else "Proceed normally"]
        }
        report_df = pd.DataFrame(report_data)
        csv_buffer = io.StringIO()
        report_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Claim Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"fraud_report_{claim_id or 'claim'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ── SAVE TO HISTORY ───────────────────────────────────
        st.session_state.history.append({
            "Timestamp": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "User ID": user_id or "N/A",
            "Policy No": policy_number or "N/A",
            "Claim ID": claim_id or "N/A",
            "Severity": incident_severity,
            "Total Claim ($)": total_claim_amount,
            "Fraud Prob (%)": fraud_prob,
            "Risk Level": risk_label,
            "Prediction": "🚨 FRAUD" if prediction==1 else "✅ LEGIT",
            "Model": mname if "All 5" not in selected_model else "Consensus"
        })
        st.session_state.total_checked += 1
        if prediction == 1:
            st.session_state.total_fraud += 1

# ════════════════════════════════════════════════════════════
# PAGE 2: BATCH PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction (CSV)":
    st.markdown("## 📁 Batch Prediction")
    st.markdown("Upload a CSV file with multiple claims to predict all at once.")

    st.markdown("### 📋 Required CSV Columns:")
    st.code("user_id, policy_number, claim_id, age, months_as_customer, incident_severity, total_claim_amount, vehicle_claim, bodily_injuries, witnesses, police_report")

    sample_data = pd.DataFrame({
        'user_id': ['USR-001','USR-002','USR-003'],
        'policy_number': ['POL-001','POL-002','POL-003'],
        'claim_id': ['CLM-001','CLM-002','CLM-003'],
        'age': [45, 32, 55],
        'months_as_customer': [12, 200, 85],
        'incident_severity': ['Total Loss','Minor Damage','Major Damage'],
        'total_claim_amount': [105000, 8500, 65000],
        'vehicle_claim': [75000, 6000, 45000],
        'bodily_injuries': [2, 0, 1],
        'witnesses': [0, 2, 1],
        'police_report': ['NO','YES','YES']
    })

    sample_csv = io.StringIO()
    sample_data.to_csv(sample_csv, index=False)
    st.download_button("📥 Download Sample CSV Template", sample_csv.getvalue(),
                       "sample_claims.csv", "text/csv", use_container_width=True)

    uploaded_file = st.file_uploader("Upload your claims CSV", type=['csv'])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown(f"✅ Loaded **{len(batch_df)}** claims")
        st.dataframe(batch_df.head())

        if st.button("🔍 Run Batch Prediction", type="primary", use_container_width=True):
            model = all_models['Random Forest']
            results = []
            progress = st.progress(0)
            for i, row in batch_df.iterrows():
                try:
                    sev = row.get('incident_severity','Minor Damage')
                    sev_map = {'Trivial Damage':3,'Minor Damage':2,'Major Damage':0,'Total Loss':1}
                    pr_map = {'YES':2,'NO':1,'?':0}
                    inp = {
                        'months_as_customer': row.get('months_as_customer',100),
                        'age': row.get('age',35),
                        'policy_state':4,'policy_csl':1,'policy_deductable':1000,
                        'policy_annual_premium':1200,'umbrella_limit':0,
                        'insured_sex':1,'insured_education_level':1,'insured_occupation':3,
                        'insured_hobbies':5,'insured_relationship':1,
                        'capital-gains':0,'capital-loss':0,
                        'incident_type':3,'collision_type':0,
                        'incident_severity': sev_map.get(sev,2),
                        'authorities_contacted':3,'incident_state':4,'incident_city':5,
                        'incident_hour_of_the_day':12,
                        'number_of_vehicles_involved':1,
                        'property_damage':1,
                        'bodily_injuries': row.get('bodily_injuries',0),
                        'witnesses': row.get('witnesses',1),
                        'police_report_available': pr_map.get(str(row.get('police_report','YES')),2),
                        'total_claim_amount': row.get('total_claim_amount',50000),
                        'injury_claim':5000,'property_claim':5000,
                        'vehicle_claim': row.get('vehicle_claim',40000),
                        'auto_make':5,'auto_model':10,'auto_year':2005
                    }
                    scaled = scaler.transform(pd.DataFrame([inp]))
                    prob = round(model.predict_proba(scaled)[0][1]*100,1)
                    pred = model.predict(scaled)[0]
                    risk, icon, _ = get_risk_level(prob)
                    results.append({
                        'User ID': row.get('user_id','N/A'),
                        'Policy No': row.get('policy_number','N/A'),
                        'Claim ID': row.get('claim_id','N/A'),
                        'Fraud Prob (%)': prob,
                        'Risk Level': f"{icon} {risk}",
                        'Prediction': '🚨 FRAUD' if pred==1 else '✅ LEGIT'
                    })
                except: pass
                progress.progress((i+1)/len(batch_df))

            results_df = pd.DataFrame(results)
            st.markdown("### 📊 Batch Results")
            st.dataframe(results_df, use_container_width=True)
            fraud_count = sum(1 for r in results if 'FRAUD' in r['Prediction'])
            st.metric("Total Claims", len(results))
            c1,c2,c3 = st.columns(3)
            c1.metric("🚨 Fraud Detected", fraud_count)
            c2.metric("✅ Legitimate", len(results)-fraud_count)
            c3.metric("Fraud Rate", f"{round(fraud_count/len(results)*100,1)}%")
            csv_out = io.StringIO()
            results_df.to_csv(csv_out, index=False)
            st.download_button("📥 Download Batch Results (CSV)", csv_out.getvalue(),
                               "batch_results.csv", "text/csv", use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3: DASHBOARD & HISTORY
# ════════════════════════════════════════════════════════════
elif page == "📊 Dashboard & History":
    st.markdown("## 📊 Session Dashboard & Claims History")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Claims Checked", st.session_state.total_checked)
    c2.metric("Fraud Detected", st.session_state.total_fraud)
    c3.metric("Legitimate", st.session_state.total_checked - st.session_state.total_fraud)
    if st.session_state.total_checked > 0:
        c4.metric("Session Fraud Rate", f"{round(st.session_state.total_fraud/st.session_state.total_checked*100,1)}%")
    st.markdown("---")
    st.markdown("### 📋 Model Performance Summary")
    perf_df = pd.DataFrame({
        'Algorithm': list(accuracies.keys()),
        'Accuracy (%)': list(accuracies.values()),
        'Best For': ['Baseline','Explainability','Overall Best','High Recall','Benchmark']
    })
    st.dataframe(perf_df, use_container_width=True)
    st.markdown("---")
    st.markdown("### 🕐 Claims History (This Session)")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv_hist = io.StringIO()
        hist_df.to_csv(csv_hist, index=False)
        st.download_button("📥 Download Full History (CSV)", csv_hist.getvalue(),
                           "claims_history.csv", "text/csv", use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.total_checked = 0
            st.session_state.total_fraud = 0
            st.rerun()
    else:
        st.info("No claims checked yet in this session. Go to Single Claim Prediction to get started.")

# ════════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ About & Model Info":
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    **Insurance Claim Fraud Detection Using Machine Learning**
    
    This application uses a trained Random Forest model to predict whether an insurance 
    claim is fraudulent based on claim details. Built as a Minor Project for B.Tech CSE 
    (Data Science), 7th Semester, 2026-27.
    """)
    st.markdown("### 🤖 Model Performance")
    perf = {'Algorithm':['Logistic Regression','Decision Tree','Random Forest','SVM','XGBoost'],
            'Accuracy':['73.18%','77.48%','88.08%','87.42%','87.42%'],
            'Precision':['75.48%','77.38%','89.24%','85.88%','85.47%'],
            'Recall':['73.12%','81.25%','88.12%','91.25%','91.88%'],
            'F1-Score':['74.29%','79.27%','88.68%','88.48%','88.55%']}
    st.dataframe(pd.DataFrame(perf), use_container_width=True)
    st.markdown("### 🔑 Top Fraud Indicators")
    fi = {'Feature':['incident_severity','insured_hobbies','vehicle_claim','total_claim_amount','collision_type'],
          'Importance':['0.2168','0.0609','0.0384','0.0373','0.0293'],
          'Meaning':['Fraudsters exaggerate severity','Lifestyle risk profile','Inflated vehicle damage','Higher amounts signal fraud','Collision pattern']}
    st.dataframe(pd.DataFrame(fi), use_container_width=True)
    st.markdown("### 👥 Project Team")
    t1,t2,t3 = st.columns(3)
    t1.info("👤 **MD Ehsan Khan**\nA023167023104")
    t2.info("👤 **Ayush Jalal**\nA023167023085")
    t3.info("👤 **Sampurna Sinha**\nA023167023208")
    st.markdown("### 🔗 Links")
    st.markdown("🌐 **Live App:** https://insurance-fraud-detection-ehsan.streamlit.app")
    st.markdown("📂 **GitHub:** https://github.com/ehsand546-netizen/insurance-fraud-detection")

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<center><small>
🛡️ Insurance Claim Fraud Detection System v2.0 | 
Random Forest Model | 88.08% Accuracy | 
B.Tech CSE (Data Science) Minor Project 2026-27 | 
Amity School of Engineering & Technology
</small></center>
""", unsafe_allow_html=True)
