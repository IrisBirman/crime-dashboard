Remove app.py
import os
import json
import requests
import joblib
import gdown
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# ---------- Config ----------
st.set_page_config(page_title="Crime Prediction Dashboard", layout="wide")

# ---------- Google Drive File IDs ----------
# אם יש לך Secrets ב-Streamlit Cloud אפשר לשים שם, אחרת ישירות כאן:
BUREAU_ID   = st.secrets.get("BUREAU_ID",   "1kmz7VFr1vDLdSiKUlhjxRwwkhNrHWyp5")
CATEGORY_ID = st.secrets.get("CATEGORY_ID", "138yKQ_pqUsV0Z-_qZ_M8FS0TAXn36SIu")

# ---------- Models loader (cached) ----------
def _looks_like_html(path, n=120):
    try:
        head = open(path, "rb").read(n).decode("utf-8", errors="ignore").lower()
        return "<html" in head or "doctype html" in head
    except Exception:
        return False

@st.cache_resource(show_spinner="Downloading/Loading models…")
def load_models():
    os.makedirs("models", exist_ok=True)
    bur_path = "models/BUREAU.joblib"
    cat_path = "models/Category.joblib"

    if not os.path.exists(bur_path):
        gdown.download(f"https://drive.google.com/uc?id={BUREAU_ID}", bur_path, quiet=False)
    if not os.path.exists(cat_path):
        gdown.download(f"https://drive.google.com/uc?id={CATEGORY_ID}", cat_path, quiet=False)

    for p in (bur_path, cat_path):
        if (not os.path.exists(p)) or os.path.getsize(p) < 1024 or _looks_like_html(p):
            raise RuntimeError(f"Downloaded file looks invalid: {p}")

    model_bureau   = joblib.load(bur_path)
    model_category = joblib.load(cat_path)
    return model_bureau, model_category

try:
    model_bureau, model_category = load_models()
except Exception as e:
    st.error("❌ טעינת מודלים נכשלה. בדקי ש־IDs נכונים וששיתפת כ־Anyone with the link.\n\nפרטים: " + str(e))
    st.stop()

# ---------- Static metadata ----------
categories = ["Property Crimes", "Assault", "Sexual Assault", "Theft", "Other"]
area_to_bureau = {
    1:'Central',2:'Central',3:'South',4:'Central',5:'South',6:'West',7:'West',8:'West',
    9:'Valley',10:'Valley',11:'Central',12:'South',13:'Central',14:'West',
    15:'Valley',16:'Valley',17:'Valley',18:'South',19:'Valley',20:'West',21:'Valley'
}

# ---------- Load LA polygons WITHOUT geopandas ----------
@st.cache_data(show_spinner="Loading LA GeoJSON…")
def load_la_geojson():
    url = "https://drive.google.com/uc?id=1pERDZzAm1aiUCEs0LSwuaSe89H9VSXCS"
    out = "LAPD_Division.geojson"
    if not os.path.exists(out):
        gdown.download(url, out, quiet=False)
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

la_geo = load_la_geojson()

# ---------- UI ----------
st.title("Crime Prediction Dashboard")
st.markdown("### Enter Crime Details:")

with st.form("prediction_form"):
    date_rptd = st.date_input("Date Reported")
    date_occ  = st.date_input("Date Occurred")
    time_occ  = st.number_input("Time Occurred (HHMM)", min_value=0, max_value=2359, step=1)

    area      = st.selectbox("Area", list(area_to_bureau.keys()))
    crm_cd    = st.text_input("Crime Code (Crm Cd)", "")

    vict_age      = st.number_input("Victim Age", min_value=0, max_value=120, step=1)
    vict_sex      = st.selectbox("Victim Sex", ["M", "F", "X", "U"])
    vict_descent  = st.text_input("Victim Descent", "")
    premis_cd     = st.text_input("Premis Cd", "")
    weapon_used   = st.text_input("Weapon Used", "")

    submitted = st.form_submit_button(" Predict")

if submitted:
    X_category = pd.DataFrame([{
        "Date Rptd": str(date_rptd),
        "DATE OCC": str(date_occ),
        "TIME OCC": time_occ,
        "AREA": area,
        "Vict Age": vict_age,
        "Vict Sex": vict_sex,
        "Vict Descent": vict_descent,
        "Weapon Used": weapon_used
    }])

    X_bureau = pd.DataFrame([{
        "Date Rptd": str(date_rptd),
        "DATE OCC": str(date_occ),
        "TIME OCC": time_occ,
        "Crm Cd": crm_cd,
        "Vict Age": vict_age,
        "Vict Sex": vict_sex,
        "Vict Descent": vict_descent,
        "Premis Cd": premis_cd,
        "Weapon Used": weapon_used
    }])

    pred_category = model_category.predict(X_category)[0]
    pred_bureau   = model_bureau.predict(X_bureau)[0]

    st.success(f"**Predicted Category:** {pred_category}")
    st.success(f"**Predicted Bureau:** {pred_bureau}")

    # ----- Map with GeoJSON (folium only) -----
    col1, col2 = st.columns([2, 1])
    with col1:
        m = folium.Map(location=[34.05, -118.25], zoom_start=10)

        for feat in la_geo.get("features", []):
            props = feat.get("properties", {})
            area_code = None
            for key in ("AREA", "area", "AREA_", "Division", "DIVISION", "id"):
                if key in props:
                    area_code = props[key]
                    break
            try:
                area_code = int(area_code)
            except Exception:
                area_code = None

            bureau_name = area_to_bureau.get(area_code, "Unknown")
            color = "red" if bureau_name == pred_bureau else "gray"

            folium.GeoJson(
                feat,
                style_function=lambda feat, col=color: {
                    "fillColor": col, "color": "black", "weight": 1, "fillOpacity": 0.5
                },
                tooltip=f"AREA: {area_code} → {bureau_name}"
            ).add_to(m)

        st_folium(m, width=700, height=500)

    with col2:
        df_cat = pd.DataFrame({"Category": categories, "Count": [1]*len(categories)})
        colors = ["gray"] * len(categories)
        if pred_category in categories:
            colors[categories.index(pred_category)] = "red"

        fig = px.pie(
            df_cat, names="Category", values="Count",
            color="Category", color_discrete_sequence=colors,
            title="Predicted Category Highlighted"
        )
        st.plotly_chart(fig, use_container_width=True)

