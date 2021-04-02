
import os
import pandas as pd
import json
import pickle
import logging 

#from azureml.core import Model
from sklearn.externals import joblib
import azureml.train.automl


def init():
    global deploy_model
    deploy_model = joblib.load('outputs/vrk_ids_model.joblib')
    
    
def transform_test_data(input_test_data):
    #load column names
    with open('ids_feature_details.json', 'r') as filehandle:
        read_dict = json.load(filehandle)
    # in dictionary keys are network_data_column_names, continious_features, symbolic_features, and
    # trained_model_column_names
    network_data_column_names_orig = read_dict['network_data_column_names']
    continious_features            = read_dict['continious_features']
    symbolic_features              = read_dict['symbolic_features']
    trained_model_column_names     = read_dict['trained_model_column_names']
    
    # for this project we don't use 'success_pred' and we are predicting the 'attack_type' so remove 'attack_type'
    # data.columns = set(network_data_column_names_orig) - set(['attack_type', 'success_pred'])
    input_test_data = pd.get_dummies(input_test_data, columns=symbolic_features)
    
    # Get missing columns in the input test data
    missing_cols = set( trained_model_column_names ) - set( input_test_data.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        input_test_data[c] = 0
    # Ensure the order of column in the test set is in the same order that in train set
    input_test_data = input_test_data[trained_model_column_names]
    
    #load scaler object which is trained with train data
    standard_scaler = load(open('ids_cont_scalerobj.pkl', 'rb'))
    
    input_test_data[continuous_features] = standard_scaler.transform(input_test_data[continuous_features])
    return input_test_data

def run(data):
    try:
        temp = json.loads(data)
        data = pd.DataFrame(temp['data'])
        transformed_test_data = transform_test_data(data)
        result = deploy_model.predict(transformed_test_data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error