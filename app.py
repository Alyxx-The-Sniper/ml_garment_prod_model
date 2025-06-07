from flask import Flask, request, render_template
import joblib
import pandas as pd
from pydantic import BaseModel, confloat, ValidationError


# Load model. `model_1.pkl` is a pipeline with:
# winsorize → median-impute → standardize → XGBRegressor.
# No need to standardize new data before prediction.
model = joblib.load("model_1.pkl")


# Flask app
app = Flask(__name__)


# Pydantic validation model: handle error of raw data to post
class InputData(BaseModel):
    incentive: confloat(ge=0.0, le=150.0)


@app.route("/", methods=["GET"])
def home():
    """GET method: browse the URL and render home.html."""
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """POST method: validate input, predict, and re-render home.html."""
    # Convert form data into a regular dict for Pydantic.
    raw = request.form.to_dict()
    try:
        validated = InputData(**raw)
    except ValidationError as e:
        # If validation fails, show errors on the same page.
        return render_template("home.html", error_message=e.errors())

    # Build a one-row DataFrame from the validated data.
    df = pd.DataFrame([validated.dict()])

    # Run the pipeline’s predict (winsorize→impute→scale under the hood).
    prediction = float(model.predict(df)[0])

    # Format the result to 3 decimal places.
    prediction_text = f"Predicted productivity: {prediction:.3f}"

    # Render home.html again, passing in the prediction text.
    return render_template("home.html", prediction_text=prediction_text)


if __name__ == "__main__":
    # Run on all interfaces, port 5000, with debug=True.
    app.run(debug=True, host="0.0.0.0", port=5000)
