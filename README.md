

# Network Intrusion Detection System (IDS)

Table of contents
=================

<!--ts-->
* [Summary](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/blob/master/starter_files/README.md#summaryy)
* [Architectural Diagram](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/tree/master/starter_files#architectural-diagram)
* [Dataset](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/tree/master/starter_files#dataset)
* [Project improvements in future](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/blob/master/starter_files/README.md#a-short-description-of-how-to-improve-project-in-future)
* [Key steps](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/tree/master/starter_files#key-steps)
* [Screen recording](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/tree/master/starter_files#screen-recording)
* [Standout suggestions](https://github.com/venkataravikumaralladi/MachineLearingOperationsProject/tree/master/starter_files#standout-suggestions)
<!--te-->

## Summary

This Capstone project is part of the Udacity Azure ML Nanodegree. The capstone project covers steps and requirements of an end-to-end machine learning pipeline in Azure. In this project, I used a NSL-KDD dataset from Kaggle to build a Intrusion Detection System (IDS) classifier. In First part of project we explore techniques that help to improve the prediction performance of a single model through hyerparameter tuning (file: hyperparameter_tuning.ipnyb) which uses HyperDrive package to tune hyperparameters of selected model and returns best model. In second part of project we use automated machine learning (AutoML) feature for automating the time consuming, iterative tasks (like feature selection,
feature generation, trying out various models) of machine learning  model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.  

Between the AutoML and Hyperdrive experiment runs, a best performing model was selected for deployment.Finally we deploy model as a webservice for online real-time scoring. Request is sent to webservice to test deployed model. Below diagrams shows overview in pictorial form.





In this I will showcase how we can use data science as a investigation tool for network traffic, here we use classfication algorithm to distinguish between normal traffic (good connections) and intrusion or attacks traffic (bad connections). A connection is a sequence of TCP packets starting and ending at some well difined times, between which data flows to and from source IP address  to a target IP address under some well defined protocol. We will create Intrusion Detection System (IDS) 


## Project Set Up and Installation

If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, you should have Azure subscription. An Azure subscription allows you to manage storage, compute, and other assets in the Azure cloud. Following software and subscriptions are required to run note book.

  * Azure subscription
  * Create Azure ML workspace. An Azure ML workspace is an Azure resource that organizes and coordinates the action of many other Azure resources
    to assis in executing and sharing machine learning workflows.
  * Install Azure ML SDK, python 3.6, Pandas, numpy, and scikit librarires.
  * In this project we use Azure Container Instance (ACI) for deployment. An Azure subscription needs to be registered to use ACI.
  
  

By fulfilling above requirements we can able to build Intrusion Detection System (IDS). IDS system is build using hyper drive concepts using Random Forest model (NetworkdataClassifier.py) implemented in hyperparameter_tuning.ipynb. Alternatively IDS model system is built using AutoML feature implemented in automl.ipynb. Best model is selected and deployed as webservice in Azure Container Instance (ACI).

## Dataset

### Overview
The dataset we use for Intrusion Detection System (IDS) is NSLKDD dataset, (A Detailed Analysis of the KDD CUP 99 Data Set,” Proceedings of the 2nd IEEE Symposium
on Computational Intelligence in Security and Defense Applications (2009): 53–58.) which is an improvement  to a classic network intrusion detection dataset used widely by security data science professionals. The original 1999 KDD Cup dataset was created for the DARPA Intrusion Detection Evaluation Program, prepared and managed by MIT Lincoln Laboratory. The data was collected over nine weeks and consists of raw tcpdump traffic in a local area network (LAN) that simulates the environment of a typical United States Air Force LAN. Some network attacks were deliberately carried out during the recording period.


### Task
Classifiers are only as good as the data used to train them, and reliably labeled data is especially important in supervised machine learning. In NSL-KDD data set
output lable is labled as normal and type of attack types. There are 38 different types of attacks. For this project I want to predict traffic as normal (0) and attack (1) types rather than types of attack. Description of features are provided in file NSL-KDD Features.xlsx

Features used for this task:

Intrinsic features:

Intrinisic features are derived from the header of the packet without looking into the payload itself, and hold the basic information about the packet. This category contains features: Duration, Protocol Type, Service, Flag, Src Bytes, Dst Bytes, Land, Wrong Fragment, and Urgent

Content features:

Content features hold information about the original packets, as they are sent in multiple  pieces rather than one. With this information, the system can access the payload.
 This category contains features: Hot, Num Failed Logins, Logged In, Num Compromised, Root Shell,  Su Attempted, Num Root, Num File Creations, Num Shells, Num Access Files, Num Outbound Cmds, Is Hot Logins, and Is Guest Login.

Time based features:

Time-based features hold the analysis of the traffic input over a two-second window and contains  information like how many connections it attempted to make to the same host. 
These features are mostly counts and rates rather than information about the content of the traffic input.  This category contains features Count, Srv Count, Serror Rate, Srv Serror Rate, Rerror Rate, Srv Rerror Rate,  Same Srv Rate, Diff Srv Rate, Srv Diff Host Rate

Host based features:

Host-based features are similar to Time-based features, except instead of analyzing over a 2-second window,  it analyzes over a series of connections made (how many requests made to the same host over x-number of connections). These features are designed to access attacks, which span longer than a two-second window time-span.
 This category contains feataures: Dst Host Count, Dst Host Srv Count, Dst Host Same Srv Rate, Dst Host Diff Srv Rate, Dst Host Same Src Port Rate, Dst Host Srv Diff Host Rate, Dst Host Serror Rate, Dst Host Srv Serror Rate, Dst Host Rerror Rate, and Dst Host Srv Rerror Rate

### Access
The dataset was downloaded from  https://www.kaggle.com/hassan06/nslkdd to my desktop and then uploaded to my Github Repo. From my Github Repo I loaded to AML work space using "TabularDatasetFactory" class function "from_delimited_files" to create a TabularDataset to represent tabular data in .CSV delimited file using AML Azure SDK. (Note: Description of data set is well described at https://www.unb.ca/cic/datasets/nsl.html)

Once Tabular data set was downloaded, it is converted to pandas data frame, cleaned it for accepting input for RandomForestClassifier by using API's get_dummies for categorial variables and Scalar object for continious variables. Here cleaning step step consists of data exploration, data cleaning, preprocessing, feature engineering. Here I checked data types (continious) for continious I used scaling features and created scalar object and stored for later to use it for testing data,  ordinal (categories) I used 'get_dummies'. This step is performed for Hyperdrive config model creation project.

In case AutomML project above clean up of data is not performed as it handled automatically by Azure AutoML module.In case of AutoML I uploaded csv file to default data store in Azure. After that I get data set pointer to data store for network training data and passed as an argument for "training_data" in AutoMLConfig API.

Once the dataset was downloaded, SDK was again used to clean and split the data into training and validation datasets, which were then stored as Pandas dataframes in memory to facilitate quick data exploration and query, and registered as AML TabularDatasets in the workspace to enable remote access by the AutoML experiment running on a remote compute cluster.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
