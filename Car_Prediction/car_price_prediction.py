import os
import pandas as pd
import datetime
import xgboost as xgb
import streamlit as st


def main():

    html_temp = """
    <h1 style='text-align:center;'>Car Price Prediction</h1>
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.json")

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("This app will help you predict your car selling price.")

    p1 = st.number_input(
        "Please enter ex-showroom price (In Lakhs)",
        min_value=2.5,
        max_value=25.0,
        step=1.0
    )

    p2 = st.number_input(
        "Please enter car driven (In Kilometers)",
        min_value=100,
        max_value=500000,
        step=100
    )

    s1 = st.selectbox(
        "Select the fuel type",
        ("Petrol", "Diesel", "CNG")
    )

    if s1 == "Petrol":
        p3 = 0
    elif s1 == "Diesel":
        p3 = 1
    else:
        p3 = 2

    s2 = st.selectbox(
        "Select the seller type",
        ("Dealer", "Individual")
    )

    if s2 == "Dealer":
        p4 = 0
    else:
        p4 = 1

    s3 = st.selectbox(
        "Select the transmission",
        ("Manual", "Automatic")
    )

    if s3 == "Manual":
        p5 = 0
    else:
        p5 = 1

    p6 = st.slider("How many owners", 0, 3)

    current_year = datetime.datetime.now().year

    years = st.number_input(
        "Car purchased year",
        min_value=1990,
        max_value=current_year,
        step=1
    )

    p7 = current_year - years

    data_new = pd.DataFrame({
        "Present_Price": [p1],
        "Kms_Driven": [p2],
        "Fuel_Type": [p3],
        "Seller_Type": [p4],
        "Transmission": [p5],
        "Owner": [p6],
        "Age": [p7]
    })

    if st.button("Predict"):
        pred = model.predict(data_new)
        st.success(
            "You can sell your car at {:.2f} lakhs".format(pred[0])
        )


if __name__ == "__main__":
    main()
