
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
import os
import json

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    if not all(os.path.exists(f) for f in ["gb_model_final.pkl", "scaler_final.pkl", "feature_columns.json"]):
        st.error("❌ One or more required files are missing: model, scaler, or feature columns.")
        st.stop()

    model = joblib.load("gb_model_final.pkl")
    scaler = joblib.load("scaler_final.pkl")
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto Slab', serif;
    }
    </style>
''', unsafe_allow_html=True)

st.image("images/intrusion_logo.jpg", use_container_width=True)

def generate_pdf_report(input_data, prediction_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Network Intrusion Detection Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction Result: {prediction_text}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Input Features:", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)
    return pdf

selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "About"],
    icons=["shield-lock", "bar-chart", "info-circle"],
    orientation="horizontal",
)

if selected == "Home":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Enter network traffic attributes below to detect whether it's **Normal** or an **Attack**.")

    # Step 1: Initialize all features with 0
    input_data = {col: 0 for col in feature_columns}

    col1, col2 = st.columns(2)

    # Step 2: Only update values for the 15 features you want users to control
    with col1:
        input_data["src_bytes"] = st.number_input("Source Bytes", min_value=0)
        input_data["same_srv_rate"] = st.slider("Same Service Rate", 0.0, 1.0, 0.5)
        input_data["flag_SF"] = st.selectbox("Flag SF", [0, 1])
        input_data["level"] = st.number_input("Level", min_value=0)
        input_data["count"] = st.number_input("Count", min_value=0)
        input_data["logged_in"] = st.selectbox("Logged In", [0, 1])
        input_data["protocol_type_icmp"] = st.selectbox("Protocol ICMP", [0, 1])

    with col2:
        input_data["dst_bytes"] = st.number_input("Destination Bytes", min_value=0)
        input_data["dst_host_same_srv_rate"] = st.slider("Host Same SRV Rate", 0.0, 1.0, 0.5)
        input_data["dst_host_srv_serror_rate"] = st.slider("Host SRV Serror Rate", 0.0, 1.0, 0.5)
        input_data["diff_srv_rate"] = st.slider("Different SRV Rate", 0.0, 1.0, 0.5)
        input_data["dst_host_srv_count"] = st.number_input("Host SRV Count", min_value=0)
        input_data["serror_rate"] = st.slider("Serror Rate", 0.0, 1.0, 0.5)
        input_data["dst_host_diff_srv_rate"] = st.slider("Host Diff SRV Rate", 0.0, 1.0, 0.5)

    if st.button("Predict Intrusion"):
        input_df = pd.DataFrame([input_data])

        # Step 3: Ensure feature alignment
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        result = "✅ Normal Traffic" if prediction == 0 else "🚨 Attack Detected"

        st.subheader(f"Prediction: {result}")

        pdf = generate_pdf_report(input_data, result)
        pdf.output("intrusion_report.pdf")

        with open("intrusion_report.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="Network_Intrusion_Report.pdf">📥 Download PDF Report</a>'
            st.markdown(download_link, unsafe_allow_html=True)

elif selected == "Dataset":
    st.title("📊 Dataset Insight")
    st.markdown("Explore the dataset used to train the network intrusion detection model.")

    try:
        df = pd.read_csv("cleaned_dataset.csv")
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df)

        if "target" in df.columns:
            st.markdown("### 🚦 Traffic Label Distribution")
            fig, ax = plt.subplots()
            sns.set_style("whitegrid")
            sns.countplot(x="target", data=df, palette="Set2", ax=ax)
            ax.set_title("Distribution of Traffic Labels")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("ℹ️ 'target' column not found. Skipping visualization.")

    except FileNotFoundError:
        st.warning("⚠️ 'cleaned_dataset.csv' file not found.")
    except pd.errors.EmptyDataError:
        st.error("🚫 The dataset file is empty.")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

elif selected == "About":
    st.title("ℹ️ About This Project")
    st.markdown('''
**Network Intrusion Detection System** is a machine learning-powered app designed to detect malicious activity 
in network traffic using Gradient Boosting.

Trained on preprocessed KDD dataset features, it provides accurate, fast, and interpretable detection of potential threats.

- 🛡️ **Model**: Gradient Boosting (F1 ≈ 0.84, AUC ≈ 0.97)  
- 👩🏽‍💻 **Developer**: Ameerah Kareem  
- 🏛️ **Institution**: Caleb University  
- 📌 **Purpose**: Data + AI for Cybersecurity Impact
''')
    st.markdown("---")
    st.markdown("<center style='color: gray;'>Made with 🔐 by Ameerah | Powered by Streamlit + Gradient Boosting</center>", unsafe_allow_html=True)
