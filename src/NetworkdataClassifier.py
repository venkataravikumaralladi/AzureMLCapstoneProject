# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:28:34 2021

@author: INVERAV
"""

# Python imports e.g., os, math...
from collections import defaultdict 
import argparse
import joblib
import os
import json
import pickle

# Third party libraries imports

# Numpy imports e.g., numpy,..
import numpy as np

# Pandas imports e.g., pandas...
import pandas as pd

# sklearn imports e.g., linear_model,..
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Azure imports e.g., Workspace,...
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# VRK: Create TabularDataset using TabularDatasetFactory

# create nsl-kdd network train data
nsl_kdd_webpath = [
                      'https://raw.githubusercontent.com/venkataravikumaralladi/AzureMLCapstoneProject/main/input/KDDTrain.csv'
                  ]

#create network analysis data set in tabular format using TabularDatasetFactory
nsl_kdd_dataset = TabularDatasetFactory.from_delimited_files(path=nsl_kdd_webpath)

class NSLKDDFeatureAnalysis:
   # class variables
   network_data_column_names = [ 
                  'duration', 'protocol_type', 'service',
                  'flag', 'src_bytes', 'dst_bytes',
                  'land', 'wrong_fragment', 'urgent',
    
            
                  'hot', 'num_failed_logins', 'logged_in',
                  'num_compromised', 'root_shell', 'su_attempted',
                  'num_root', 'num_file_creations', 'num_shells',
                  'num_access_files', 'num_outbound_cmds', 'is_hot_login',
                  'is_guest_login',
    
                 
                  'count', 'srv_count', 'serror_rate',
                  'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                 
                  'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                  'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                  'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                  'dst_host_srv_rerror_rate',
    
                   'attack_type',
                   'success_pred' ]
   
   def __init__(self, data):
      self.train_data = data
      
   def clean_data(self):
      
      train_df = self.train_data.to_pandas_dataframe().dropna()
      train_df.columns = NSLKDDFeatureAnalysis.network_data_column_names
          
	  # For this analysis we drop "success_pred" column
      train_df.drop('success_pred', axis=1, inplace=True)
    
	  # Drop attack type in training data which is to be predicted.
      train_X = train_df.drop("attack_type", axis=1)
      train_Y = train_df['attack_type']

	  # convert categorical types to dummy variables.    
      feature_type_to_names_mapping = defaultdict(list)
		
      with open('KDDDataFeatureNamesToTypes.txt', 'r') as f:
          #read from line 1. skip line 0 as in given file we have attack names here which we don't need.
          for line in f.readlines()[1:]:
              name, nature = line.strip()[:-1].split(': ')
              feature_type_to_names_mapping[nature].append(name)
		
	  # Generate dummy variables for categorical types
      train_data_X = pd.get_dummies(train_X, columns=feature_type_to_names_mapping['symbolic'], drop_first=False)
		
      # standarize continious feature
      continuous_features = feature_type_to_names_mapping['continuous']
      symbolic_features   = feature_type_to_names_mapping['symbolic']
      strd_scalar_continious = StandardScaler().fit(train_data_X[continuous_features])
	  # Standardize training data
      train_data_X[continuous_features] = strd_scalar_continious.transform(train_data_X[continuous_features])

	  # we build binary classifier for this.
      train_Y = train_Y.apply(lambda x: 0 if x == 'normal' else 1)  

      ids_columns_details_dict = {
                      "orig_network_data_column_names": NSLKDDFeatureAnalysis.network_data_column_names,
                      "continious_features" : continuous_features,
                      "symbolic_names" : symbolic_features,
                      "trained_model_column_names": train_data_X.columns.tolist()
                      }
		
      # write feature engineering column detials to ids_feature_cols
      os.makedirs('outputs', exist_ok=True)
      with open('outputs/ids_feature_details.json', 'w') as filehandle:
          json.dump(ids_columns_details_dict, filehandle)
      
          
      # write standard scalar object created with trained object for later use for test data.
      pickle.dump(strd_scalar_continious, open('outputs/ids_cont_scalerobj.pkl', 'wb'))
            
      return train_data_X, train_Y

   


def main():
    
    run = Run.get_context()
    
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', default='gini', help="The function to measure the quality of a split.")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree")

    args = parser.parse_args()

    run.log("Criteria for split:", args.criterion)
    run.log("Max depth:", args.max_depth)
    
    # VRK: Data cleaning step
    nsl_data_analysis = NSLKDDFeatureAnalysis(nsl_kdd_dataset)
    x, y = nsl_data_analysis.clean_data()
    
    # VRK: Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    
    decisiontree_attack_classifier = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    decisiontree_attack_classifier.fit(x_train, y_train)
    
    accuracy = decisiontree_attack_classifier.score(x_test, y_test)
        
    #VRK:Save the model.
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(decisiontree_attack_classifier, 'outputs/vrk_ids_model.joblib')
    run.log("Accuracy", np.float(accuracy))
    return


if __name__ == '__main__':
    
    main()    

	

    

    
