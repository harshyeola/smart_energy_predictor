import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Industrial Energy Predictor", page_icon="⚡", layout="centered")

# Load and train models cached for performance
@st.cache_data
def load_and_train_models():
    np.random.seed(42)
    n_samples = 1000

    lagging_power = np.random.uniform(5, 75, n_samples)
    leading_power = np.random.uniform(0, 35, n_samples)
    co2 = np.random.uniform(0, 0.05, n_samples)
    power_factor = np.random.uniform(60, 100, n_samples)

    energy_kwh = (lagging_power * 1.2) - (leading_power * 0.4) + (co2 * 100) + (power_factor * 0.5) + np.random.normal(0, 5, n_samples)

    df = pd.DataFrame({
        'Lagging_Power': lagging_power,
        'Leading_Power': leading_power,
        'CO2': co2,
        'Power_Factor': power_factor,
        'Energy_kWh': energy_kwh
    })

    X = df[['Lagging_Power', 'Leading_Power', 'CO2', 'Power_Factor']]
    y = df['Energy_kWh']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5)
    }

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_models[name] = model
        metrics[name] = {
            'MAE': round(mean_absolute_error(y_test, preds), 2),
            'R2': round(r2_score(y_test, preds), 3)
        }
        
    return trained_models, metrics

trained_models, metrics = load_and_train_models()

# UI Layout
st.title("⚡ Smart Industrial Energy Predictor")
st.markdown("Predict the energy consumption of industrial machinery using Machine Learning algorithms.")

st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select ML Model", list(trained_models.keys()))

st.subheader("Enter Plant Telemetry Parameters:")
col1, col2 = st.columns(2)

with col1:
    lagging = st.number_input("Lagging Current Reactive Power (kvarh)", value=35.5, step=0.1)
    co2 = st.number_input("CO2 Emission (ppm)", value=0.02, step=0.001, format="%.3f")

with col2:
    leading = st.number_input("Leading Current Reactive Power (kvarh)", value=12.0, step=0.1)
    pf = st.number_input("Power Factor (%)", value=85.0, step=0.1)

if st.button("Predict Energy Consumption", type="primary"):
    input_data = np.array([[lagging, leading, co2, pf]])
    model = trained_models[selected_model]
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted Energy Output: {prediction:.2f} kWh")
    
    model_metrics = metrics[selected_model]
    st.info(f"**Model Performance Metrics:** R² Score: `{model_metrics['R2']}` | Mean Absolute Error (MAE): `{model_metrics['MAE']}`")
