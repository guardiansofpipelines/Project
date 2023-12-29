import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_and_read_data(file_path):
    # Read data from a CSV file using pandas
    return pd.read_csv(file_path)

def preprocess_dataframe(dataframe):
    # Use MinMaxScaler to normalize 'SensorValue'
    scaler = preprocessing.MinMaxScaler()
    scaled_values = scaler.fit_transform(dataframe['SensorValue'].values.reshape(-1, 1))
    dataframe['SensorValue'] = scaled_values

    # Drop rows with missing values and unnecessary columns
    dataframe = dataframe.dropna() 
    dataframe = dataframe.drop(columns=['DeviceID', 'SensorID'])

    # Convert 'RecordTime' to datetime format and extract additional features
    dataframe['RecordTime'] = pd.to_datetime(dataframe['RecordTime'])
    dataframe['hour_of_day'] = dataframe['RecordTime'].dt.hour
    dataframe['day_of_week'] = dataframe['RecordTime'].dt.dayofweek
    dataframe['year'] = dataframe['RecordTime'].dt.year

    return dataframe

def split_dataframe(dataframe, split_ratio):
    # Split the dataframe into features and labels, and perform one-hot encoding
    features = dataframe.drop('SensorValue', axis=1)
    label = dataframe['SensorValue']
    features = pd.get_dummies(features)  # one-hot encoding categorical variables

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=split_ratio, random_state=42)

    # Concatenate features and labels to create new dataframes
    train_dataframe = pd.concat([X_train, y_train], axis=1)
    test_dataframe = pd.concat([X_test, y_test], axis=1)

    return train_dataframe, test_dataframe

def perform_data_processing():
    print("Performing generic data processing")

    # Load the file paths for training and testing output
    train_output_file = sys.argv[1]
    test_output_file = sys.argv[2]
    
    # Load raw data from a CSV file
    file_path = "data/raw/sensor_data.csv"
    dataframe = load_and_read_data(file_path)
    
    # Preprocess sensor values
    dataframe = preprocess_dataframe(dataframe)
    
    # Split data into training and testing sets
    split_ratio = yaml.safe_load(open('src/generic_params.yaml'))['prepare']['split']
    train_dataframe, test_dataframe = split_dataframe(dataframe, split_ratio)
    
    # Write the processed data to CSV files
    train_dataframe.to_csv(train_output_file, index=False)
    test_dataframe.to_csv(test_output_file, index=False)

    return

# Runtime call:
# python3 src/generic_prepare.py data/generic_prepared/train/train.csv data/generic_prepared/test.csv

if __name__ == "__main__":
    perform_data_processing()
