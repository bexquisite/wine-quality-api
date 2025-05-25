import joblib
import pandas as pd

# Load model
model = joblib.load("model/wine_quality_model.pkl")

# Load sample data
df = pd.read_csv("data/winequality-red.csv", sep=";")
sample_input = df.iloc[0].drop("quality").to_dict()

def test_model_prediction():
    prediction = model.predict(pd.DataFrame([sample_input]))[0]
    assert 3 <= prediction <= 8  # Wine quality is between 3 and 8

def test_flask_app_exists():
    try:
        from app.app import app
        assert app
    except ImportError:
        assert False, "Flask app not found"
