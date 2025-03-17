from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("fish_model.pkl")

@app.route('/')
def home():
    return "Welcome to the Fish Weight Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert JSON into DataFrame
        df = pd.DataFrame(data, index=[0])

        # Make prediction
        prediction = model.predict(df)

        # Return the prediction
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
