import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import json

# PAGE CONFIG 
st.set_page_config(
    page_title="NYC Taxi - sprawdź napiwek $$",
    page_icon="🚕",
    layout="wide"
)

# LOAD MODELS
with open("xgb_classifier_model.pkl", "rb") as f:
    tip_classifier = pickle.load(f)

with open("xgb_regression_model.pkl", "rb") as f:
    tip_regressor = pickle.load(f)

# STYLES
st.markdown("""
<style>
.stApp {
    background-color: #111;
    color: #f7c600;
}
h1, h2, h3 {
    color: #f7c600;
}          
.card {
    background: #f7c600;
    color: #111;
    padding: 20px;
    border-radius: 15px;
    font-weight: bold;
}
.stButton>button {
    background-color: #f7c600;
    color: black;
    font-weight: bold;
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    background-color: #000;
    border-right: 4px solid #f7c600;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🚕 NYC Taxi ML projekt")
st.sidebar.markdown("**Sprawdź czy i jaki napiwek dostaniesz** 💵")

# HEADER 
st.markdown("# 🚕 NYC TAXI - ")
st.markdown("### 💸 Czy klient zostawi napiwek? W jakiej wysokości?")

# INPUTS
st.markdown("## 📥 Wprowadź szczegółowe dane przejazdu")

col1, col2, col3 = st.columns(3)

with col1:
    fare_amount = st.number_input("Opłata podstawowa ($)", 3.0, 300.0, 18.5)
    ratecode = st.selectbox('Kod taryfy', [1,2,3,4,5,6])
    congestion_surcharge = st.number_input("Congestion surcharge ($)", 0.0, 10.0, 2.75)
    
with col2:
    extra = st.number_input("Extra charges ($)", 0.0, 20.0, 0.5)
    airport_fee = st.number_input("Airport fee ($)", 0.0, 10.0, 0.0)
    payment_type = st.selectbox(
        "Typ płatności",
        ["Karta", "Gotówka", "Brak opłaty", "Spór"]
    )
    payment_type_map = {"Karta":1, "Gotówka":2, "Brak opłaty":3, "Spór":4}
    payment_type_int = payment_type_map[payment_type]

with col3:
    is_night = st.selectbox('Przejazd w godzinach nocnych', ["Nie", "Tak"])
    is_night_map = {"Nie":0, "Tak":1}
    is_night_int = is_night_map[is_night]
    
    is_short_trip = st.selectbox('Typ przejazdu', ["Krótki", "Długi"])
    is_short_trip_map = {"Krótki":0, "Długi":1}
    is_short_trip_int = is_short_trip_map[is_short_trip]
    
    improvement_surcharge = st.number_input("improvement_surcharge ($)", 0.0, 1.0, 0.1)
    

# FEATURE ENGINEERING
fare_extras_sum = congestion_surcharge + extra + airport_fee + improvement_surcharge 

X = np.array([[
    payment_type_int,     # 0
    ratecode,             # 1
    congestion_surcharge, # 2
    extra,                # 3
    airport_fee,          # 4
    fare_extras_sum,      # 5
    is_night_int,         # 6
    is_short_trip_int,    # 7
    fare_amount,          # 8
    improvement_surcharge # 9
]], dtype=float)

# PREDICTION
if st.button("💵 Sprawdź czy będzie extra cash"):
    tip_given = tip_classifier.predict(X)[0]

    st.markdown("## 🔮 Wynik przewidywania modelu:")

    if tip_given == 1:
        tip_amount = tip_regressor.predict(X)[0]
        tip_amount = max(0, tip_amount) 
        st.markdown(
            f"""
            <div class="card">
            ✅ Klient zostawi napiwek <br><br>
            💵 Spodziewana wysokość napiwku: <span style="font-size:32px">${tip_amount:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="card">
            ❌ Klient nie zostawi napiwku <br><br>
            Może następnym razem 🚕
            </div>
            """,
            unsafe_allow_html=True
        )
