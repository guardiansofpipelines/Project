import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_and_read_data(file_path):
    # Read data from a CSV file using pandas
    return pd.read_csv(file_path)

def preprocess_dataframe(dataframe):
    # Use MinMaxScaler to normalize 'Reading'
    scaler = preprocessing.MinMaxScaler()
    scaled_values = scaler.fit_transform(dataframe['Reading'].values.reshape(-1, 1))
    dataframe['Reading'] = scaled_values

    # Drop rows with missing values and unnecessary columns
    dataframe = dataframe.dropna() 
    dataframe = dataframe.drop(columns=['Machine_ID', 'Sensor_ID'])

    # Convert 'Timestamp' to datetime format and extract additional features
    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
    dataframe['hour'] = dataframe['Timestamp'].dt.hour
    dataframe['day'] = dataframe['Timestamp'].dt.dayofweek
    dataframe['year'] = dataframe['Timestamp'].dt.year

    return dataframe


def split_dataframe(dataframe, split_ratio):
    # Assuming 'Reading' is the target variable
    label = dataframe['Reading']

    # Drop the target variable and other unnecessary columns
    features = dataframe.drop(['Reading'], axis=1)

    # Perform one-hot encoding for categorical variables
    features = pd.get_dummies(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=split_ratio, random_state=42)

    # Concatenate features and labels to create new dataframes
    train_dataframe = pd.concat([X_train, y_train], axis=1)
    test_dataframe = pd.concat([X_test, y_test], axis=1)

    return train_dataframe, test_dataframe


def perform_data_processing():
    print("Performing generic data processing")

    # Load the file paths for training and testing output
    train_output_file = sys.argv[1]+"train.csv"
    test_output_file = sys.argv[2]+"test.csv"
    
    # Load raw data from a CSV file
    input_file_path = "data/dummy_data/dummy_sensor_data.csv"
    dataframe = load_and_read_data(input_file_path)
    
    # Preprocess sensor values
    dataframe = preprocess_dataframe(dataframe)
    
    # Split data into training and testing sets
    split_ratio = yaml.safe_load(open('src/parameters.yaml'))['prepare']['split']
    train_dataframe, test_dataframe = split_dataframe(dataframe, split_ratio)
    
    # Write the processed data to CSV files
    train_dataframe.to_csv(train_output_file, index=False)
    test_dataframe.to_csv(test_output_file, index=False)

    return

# Runtime call:
# python3 src/generic_prepare.py data/generic_prepared/train/train.csv data/generic_prepared/test.csv

if __name__ == "__main__":
    perform_data_processing()
