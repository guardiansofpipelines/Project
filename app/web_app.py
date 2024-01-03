from flask import Flask, request, jsonify, render_template
import mlflow.sklearn
import pandas as pd
import mlflow
from io import StringIO

# Load the MLflow model
model = mlflow.sklearn.load_model(f"random_forest_model")

# Create the Flask application
app = Flask(__name__)

# Define a route for the root URL
@app.route("/")
def home():
    return render_template("index.html")

# Define a route for prediction that handles both GET and POST requests
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            # Check if the request data is a file upload
            if "file" in request.files:
                # Read CSV data from the uploaded file
                file = request.files["file"]
                df = pd.read_csv(file)
            else:
                # Get the input data from the request as JSON
                data = request.get_json()

                # Preprocess the input data
                df = pd.DataFrame(data)
            
            # drop column timestamp and reading
            df = df.drop(['Timestamp', 'Reading'], axis=1)
            # Make predictions using the loaded model
            predictions = model.predict(df)

            # Return the predictions as a JSON response
            return jsonify(predictions.tolist())
        elif request.method == "GET":
            # Handle GET requests (optional)
            return "GET request received for /predict endpoint. This endpoint supports POST requests for predictions."
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
