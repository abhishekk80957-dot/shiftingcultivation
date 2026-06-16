import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Shifting Cultivation Risk Prediction",
    page_icon="🌳",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1441974231531-c6227db76b6e");
    background-size: cover;
    background-attachment: fixed;
}

.main-title {
    text-align: center;
    color: white;
    background-color: rgba(0,100,0,0.85);
    padding: 15px;
    border-radius: 12px;
}

.section-header {
    color: white;
    background-color: rgba(34,139,34,0.85);
    padding: 10px;
    border-radius: 10px;
}

.metric-card {
    background-color: rgba(255,255,255,0.85);
    padding: 15px;
    border-radius: 10px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# TITLE
# ======================================

st.markdown(
    '<h1 class="main-title">🌳 Shifting Cultivation Risk Prediction System</h1>',
    unsafe_allow_html=True
)

# ======================================
# LOAD DATASET
# ======================================

df = pd.read_excel("shifting_cultivation.xlsx")

# ======================================
# DATASET OVERVIEW
# ======================================

st.markdown(
    '<h2 class="section-header">📊 Dataset Overview</h2>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Features", len(df.columns))

with col3:
    st.metric("Target Classes", df['risk_class'].nunique())

# ======================================
# ENCODING
# ======================================

state_encoder = LabelEncoder()
zone_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

df['state'] = state_encoder.fit_transform(df['state'])
df['agro_zone'] = zone_encoder.fit_transform(df['agro_zone'])
df['risk_class'] = risk_encoder.fit_transform(df['risk_class'])

# ======================================
# FEATURES & TARGET
# ======================================

X = df.drop('risk_class', axis=1)

y = df['risk_class']

# ======================================
# SCALING
# ======================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ======================================
# MODEL TRAINING
# ======================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_scaled, y)

# ======================================
# INPUT SECTION
# ======================================

st.markdown(
    '<h2 class="section-header">📋 Enter Input Values</h2>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    state_name = st.selectbox(
        "State",
        state_encoder.classes_
    )

    year = st.number_input(
        "Year",
        min_value=2020,
        max_value=2050,
        value=2030
    )

    agro_zone_name = st.selectbox(
        "Agro Zone",
        zone_encoder.classes_
    )

    jhum_cycle_years = st.number_input(
        "Jhum Cycle Years",
        value=5.0
    )

    jhum_area_km2 = st.number_input(
        "Jhum Area (km²)",
        value=100.0
    )

    fallow_land_km2 = st.number_input(
        "Fallow Land (km²)",
        value=50.0
    )

    forest_cover_km2 = st.number_input(
        "Forest Cover (km²)",
        value=500.0
    )

with col2:

    annual_rainfall_mm = st.number_input(
        "Annual Rainfall (mm)",
        value=2000.0
    )

    tribal_population_thousands = st.number_input(
        "Tribal Population (Thousands)",
        value=100.0
    )

    population_pressure_index = st.number_input(
        "Population Pressure Index",
        value=5.0
    )

    crop_yield_kg_per_ha = st.number_input(
        "Crop Yield (kg/ha)",
        value=2000.0
    )

    fertiliser_use_kg_per_ha = st.number_input(
        "Fertiliser Use (kg/ha)",
        value=100.0
    )

    forest_loss_km2 = st.number_input(
        "Forest Loss (km²)",
        value=20.0
    )

    open_forest_conversion_pct = st.number_input(
        "Open Forest Conversion (%)",
        value=10.0
    )

# ======================================
# PREDICTION
# ======================================

if st.button("🔍 Predict Risk Level"):

    state_encoded = state_encoder.transform([state_name])[0]
    zone_encoded = zone_encoder.transform([agro_zone_name])[0]

    input_data = pd.DataFrame({

        'state':[state_encoded],
        'year':[year],
        'agro_zone':[zone_encoded],
        'jhum_cycle_years':[jhum_cycle_years],
        'jhum_area_km2':[jhum_area_km2],
        'fallow_land_km2':[fallow_land_km2],
        'forest_cover_km2':[forest_cover_km2],
        'annual_rainfall_mm':[annual_rainfall_mm],
        'tribal_population_thousands':[tribal_population_thousands],
        'population_pressure_index':[population_pressure_index],
        'crop_yield_kg_per_ha':[crop_yield_kg_per_ha],
        'fertiliser_use_kg_per_ha':[fertiliser_use_kg_per_ha],
        'forest_loss_km2':[forest_loss_km2],
        'open_forest_conversion_pct':[open_forest_conversion_pct]

    })

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    risk = risk_encoder.inverse_transform(prediction)[0]

    st.markdown(
        '<h2 class="section-header">📈 Prediction Result</h2>',
        unsafe_allow_html=True
    )

    if str(risk).lower() == "high":

        st.markdown("""
        <div style="
        background-color:#ff3333;
        color:white;
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:32px;
        font-weight:bold;">
        🔴 HIGH DEFORESTATION RISK
        </div>
        """, unsafe_allow_html=True)

    elif str(risk).lower() == "medium":

        st.markdown("""
        <div style="
        background-color:#8000ff;
        color:white;
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:32px;
        font-weight:bold;">
        🟣 MEDIUM DEFORESTATION RISK
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background-color:#00cc66;
        color:white;
        padding:25px;
        border-radius:15px;
        text-align:center;
        font-size:32px;
        font-weight:bold;">
        🟢 LOW DEFORESTATION RISK
        </div>
        """, unsafe_allow_html=True)

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.success(
    "AI-Based Shifting Cultivation & Deforestation Risk Prediction System"
)