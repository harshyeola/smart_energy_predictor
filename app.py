from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

# 1. Generate Data (Matches UCI Steel Industry Dataset structure)
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

# 2. Train Models
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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = "Linear Regression"
    model_metrics = None
    inputs = {
        "lagging": 35.5,
        "leading": 12.0,
        "co2": 0.02,
        "pf": 85.0
    }

    if request.method == "POST":
        try:
            inputs["lagging"] = float(request.form["lagging"])
            inputs["leading"] = float(request.form["leading"])
            inputs["co2"] = float(request.form["co2"])
            inputs["pf"] = float(request.form["pf"])
            selected_model = request.form["model"]

            input_data = np.array([[inputs["lagging"], inputs["leading"], inputs["co2"], inputs["pf"]]])
            model = trained_models[selected_model]
            prediction = round(model.predict(input_data)[0], 2)
            model_metrics = metrics[selected_model]
        except ValueError:
            prediction = "Invalid input. Please enter numeric values."

    return render_template(
        "index.html",
        models=list(models.keys()),
        selected_model=selected_model,
        prediction=prediction,
        metrics=model_metrics,
        inputs=inputs
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
