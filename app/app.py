from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/wine_quality_model.pkl")

# Initialize Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON input
    data = request.get_json()
    # Convert to DataFrame
    df = pd.DataFrame([data])
    # Make prediction
    prediction = model.predict(df)[0]
    # Return JSON response
    return jsonify({"predicted_quality": float(prediction)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
