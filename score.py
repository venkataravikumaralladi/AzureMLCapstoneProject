
import os
import pandas as pd
import json
import pickle
import logging 
import joblib



def init():
    global deploy_model
    global read_dict
    global standard_scaler
    
    #Get the path where the deployed model can be found
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'outputs')
    print("Model path ", model_path)
    #load models
    deploy_model = joblib.load(model_path + '/vrk_ids_model.joblib')
    
    #load column names
    with open(model_path +'/ids_feature_details.json', 'r') as filehandle:
        read_dict = json.load(filehandle)
    
    #load scaler object which is trained with train data
    standard_scaler = pickle.load(open(model_path + '/ids_cont_scalerobj.pkl', 'rb'))
    
    
def transform_test_data(input_test_data):
    
    # in dictionary keys are network_data_column_names, continious_features, symbolic_features, and
    # trained_model_column_names
    #print("Input test data shape ", input_test_data.shape)
     
    network_data_column_names_orig = read_dict['orig_network_data_column_names']
    continious_features            = read_dict['continious_features']
    symbolic_features              = read_dict['symbolic_names']
    trained_model_column_names     = read_dict['trained_model_column_names']
    
    #print("continious_features ", continious_features)
    #print("symbolic_features ", symbolic_features)
    #print("trained_model_column_names ", trained_model_column_names)
    
    
    # for this project we don't use 'success_pred' and we are predicting the 'attack_type' so remove 'attack_type'
    # data.columns = set(network_data_column_names_orig) - set(['attack_type', 'success_pred'])
    input_test_data = pd.get_dummies(input_test_data, columns=symbolic_features)
    
    #print("Input test data shape after get dummies ", input_test_data.shape)
    
    # Get missing columns in the input test data
    missing_cols = set( trained_model_column_names ) - set( input_test_data.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        input_test_data[c] = 0
    # Ensure the order of column in the test set is in the same order that in train set
    input_test_data = input_test_data[trained_model_column_names]
    
    #print("Input test data shape added get dummies ", input_test_data.shape)
        
    #input_test_data[continious_features] = standard_scaler.transform(input_test_data[continious_features])
    input_test_data[continious_features] = standard_scaler.transform(input_test_data[continious_features])
    
    #print("Input test data shape after apply scalar ", input_test_data.shape)
    
    return input_test_data

def run(data):
    try:
        temp = json.loads(data)
        data = pd.DataFrame(temp['data'])
        transformed_test_data = transform_test_data(data)
        result = deploy_model.predict(transformed_test_data)
        print("Result is ", result)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        prinrt("Error occured ", error)
        return error
