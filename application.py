import streamlit as st
import mlflow.pyfunc
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow.xgboost
import requests

st.set_page_config(
    page_title="Amazon Delivery Time Prediction", page_icon="🚚", layout="centered"
)

st.title("Amazon Delivery Time Prediction")
st.subheader("Fill the details below to predict delivery time in hours.")

# Load model from MLflow
# loaded_model = mlflow.pyfunc.load_model("models:/XGBoost_RegModel/1")
"""
if I deployed the model using MLflow Model Registry, I would use the above line to load the model directly from there.
"""
# previously used for loading model from MLflow Model Registry
# Now using direct XGBoost model loading
# Load the trained XGBoost model
# loaded_model = xgb.XGBRegressor()
# loaded_model.load_model(
#     r"mlruns\190647272504744062\models\m-0cc85a7892ec4215bf3cfacea01ea75b\artifacts\model.xgb"
# )

# r is used for raw string to handle backslashes in Windows paths


## https://drive.google.com/file/d/1ImSC4VeSzzeOa89x3FHd2hqVLjNqTgig/view?usp=sharing

file_id = "1ImSC4VeSzzeOa89x3FHd2hqVLjNqTgig"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

# response = requests.get(download_url)
# with open("model.xgb", "wb") as f:
#     f.write(response.content)

loaded_model = xgb.Booster()
loaded_model.load_model("model.xgb")

df = pd.read_csv("Cleaned_amazon_delivery.csv")
le = LabelEncoder()

# Input Form

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        Agent_Age = st.number_input("Agent Age", min_value=18, max_value=65, value=30)
        Agent_Rating = st.slider(
            "Agent Rating", min_value=1.0, max_value=5.0, value=4.5
        )
        Traffic = st.selectbox("Traffic Condition", df["Traffic"].unique())
        Weather = st.selectbox("Weather Condition", df["Weather"].unique())
        Is_RushHour = st.radio("Is it Rush Hour?", ["YES", "NO"])
        Is_Weekend = st.radio("Is it Weekend?", ["YES", "NO"])
        Is_RushHour = 1 if Is_RushHour == "YES" else 0
        Is_Weekend = 1 if Is_Weekend == "YES" else 0

    with col2:
        Order_Hour = st.number_input(
            "Order Hour (0-23)", min_value=0, max_value=23, value=12
        )
        Area = st.selectbox("Area Type", df["Area"].unique())
        Vehicle = st.selectbox("Vehicle Type", df["Vehicle"].unique())
        Category = st.selectbox("Product Category", df["Category"].unique())
        Distance_km = st.number_input("Distance (km)", min_value=0.0, value=5.0)

    submitted = st.form_submit_button("Predict Delivery Time")

if submitted:
    # Prepare input data
    new_data = pd.DataFrame(
        [
            {
                "Agent_Age": Agent_Age,
                "Agent_Rating": Agent_Rating,
                "Traffic": le.fit(df["Traffic"]).transform([Traffic])[0],
                "Weather": le.fit(df["Weather"]).transform([Weather])[0],
                "Is_RushHour": Is_RushHour,
                "Is_Weekend": Is_Weekend,
                "Order_Hour": Order_Hour,
                "Area": le.fit(df["Area"]).transform([Area])[0],
                "Vehicle": le.fit(df["Vehicle"]).transform([Vehicle])[0],
                "Category": le.fit(df["Category"]).transform([Category])[0],
                "Distance_km": Distance_km,
            }
        ]
    )
    
    # Convert to DMatrix
    dmatrix_data = xgb.DMatrix(new_data)
    
    # Make prediction
    with st.spinner("Predicting... ⏳"):
        answer = loaded_model.predict(dmatrix_data)[0]
        st.toast("Prediction successful!", icon="✅")  # Popup toast messages

    st.markdown(
        f"""
        <h1 style='text-align: center; color: #2E8B57;'>
            🚚 Estimated Delivery Time: {answer:.2f} hour
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.balloons()


# deplyed url = https://predictdelivery-time.streamlit.app/