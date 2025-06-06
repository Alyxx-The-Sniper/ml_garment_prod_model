from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pydantic import BaseModel, confloat, ValidationError

# Load model
model = joblib.load("model_1.pkl") 
# model_1.pkl is pipeline 
# with [ winsorize → median‐impute → standardize ]  →  XGBRegressor
# no need to standadize new data for prediction here

# Flask app
app = Flask(__name__)

# Pydantic validation model
# handle error of raw data to post
class InputData(BaseModel):
    incentive: confloat(ge=0.0, le=150.0)

@app.route('/', methods=['GET']) # GET method > browse the url (html)
def home():
    return render_template('home.html') # the flask render_template will look for templates folder and home.html file

@app.route('/predict', methods=['POST']) # POST method > submit data
def predict():
    # Convert the form data into a regular dict,
    # so we can feed it into Pydantic.
    raw = request.form.to_dict()  
    try:
        validated = InputData(**raw)
    except ValidationError as e:
        # If validation fails, show the errors on the same page
        return render_template("home.html", error_message=e.errors())

    # Build a one-row DataFrame from the validated data
    df = pd.DataFrame([validated.dict()])

    # Run the pipeline’s predict (it will winsorize→impute→scale under the hood)
    prediction = float(model.predict(df)[0])

    # Format the result to 3 decimal places
    prediction_text = f"Predicted productivity: {prediction:.3f}"

    # Render home.html again, passing in the prediction_text
    return render_template("home.html", prediction_text=prediction_text)


if __name__ == '__main__':
    # Run on all interfaces, port 5000, with debug=True (auto-reload + interactive debugger)
    app.run(debug=True, host='0.0.0.0', port=5000)

#  if request.get_json()
# @app.route('/predict', methods=['POST'])
# def api_predict():
#     try:
#         data = request.get_json()
#         validated = InputData(**data)
#         df = pd.DataFrame([validated.dict()])
#         prediction = float(model.predict(df)[0])
#         return jsonify({'predicted_productivity': round(prediction, 3)})
#     except ValidationError as e:
#         return jsonify({'error': e.errors()}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500