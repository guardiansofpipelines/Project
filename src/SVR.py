import shutil
import os
import mlflow
import mlflow.sklearn
from sklearn.svm import SVR  # Add this line
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import sys


# Set tracking URI
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load dataset
def load_data(train_file_path, test_file_path):
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)

    # Drop unnecessary columns
    if 'Unnamed: 0' in train.columns:
        train = train.drop(columns=['Unnamed: 0'], axis=1)

    if 'Timestamp' in train.columns:
        train = train.drop(columns=['Timestamp'], axis=1)

    if 'Unnamed: 0' in test.columns:
        test = test.drop(columns=['Unnamed: 0'], axis=1)

    if 'Timestamp' in test.columns:
        test = test.drop(columns=['Timestamp'], axis=1)

    # Assuming 'Reading' is the target variable
    label_column = 'Reading'
    features_train = train.drop(label_column, axis=1)
    label_train = train[label_column]

    features_test = test.drop(label_column, axis=1)
    label_test = test[label_column]

    return features_train, features_test, label_train, label_test


def train(X_train, X_test, y_train, y_test):
    # Start an MLflow run
    with mlflow.start_run() as run:

        # Train a Support Vector Machine regressor (replace RandomForestRegressor with SVR)
        model = SVR(kernel='linear', C=1.0)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log model parameters and metrics using MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Remove the existing 'svm_model' directory if it exists
        if os.path.exists("svm_model"):
            shutil.rmtree("svm_model")

        mlflow.set_tag("svm model", mlflow.active_run().info.run_id)

        # Save the model with MLflow
        mlflow.sklearn.log_model(model, "svm_model")

        # Print evaluation metrics
        print("Mean Squared Error (MSE):", mse)
        print("Mean Absolute Error (MAE):", mae)
        print("R2 Score:", r2)
        
        # append metrics into a csv file with its column name
        df = pd.DataFrame([[mse, mae, r2]], columns=['mse', 'mae', 'r2'])
        df.to_csv('metrics.csv', mode='a', header=False, index=False)
        

def main():
    # Set the experiment name
    mlflow.set_experiment("support_vector_machine")

    # Check if there are enough command line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <train_file_path> <test_file_path>")
        sys.exit(1)

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    # Load data
    X_train, X_test, y_train, y_test = load_data(train_file_path, test_file_path)

    # Train and log the model
    train(X_train, X_test, y_train, y_test)
    
    print("Done")

if __name__ == "_main_":
    main()
