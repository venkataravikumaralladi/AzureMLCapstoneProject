

# Network Intrusion Detection System (IDS)

Table of contents
=================

<!--ts-->
* [Summary](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#summary)
* [Project Set Up and Installation](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#project-set-up-and-installation)
* [Dataset](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#dataset)
* [Automated ML](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#automated-ml)
* [Hyperparameter tuning](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#hyperparameter-tuning)
* [Model deployment](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#model-deployment)
* [Screen recording](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#screen-recording)
* [Standout suggestions](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#standout-suggestions)
* [Future work](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject#future-work)
<!--te-->

## Summary

This Capstone project is part of the Udacity Azure ML Nanodegree. The capstone project covers steps and requirements of an end-to-end machine learning pipeline in Azure. In this project, I used a NSL-KDD dataset from Kaggle to build a Intrusion Detection System (IDS) classifier. In First part of project we explore techniques that help to improve the prediction performance of a single model through hyerparameter tuning (file:`hyperparameter_tuning.ipnyb`) which uses HyperDrive package to tune hyperparameters of selected model and returns best model. In second part of project we use automated machine learning (AutoML) feature for automating the time consuming, iterative tasks (like feature selection, feature generation, trying out various models) of machine learning  model development (`automl.ipnyb`). It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.  

Between the AutoML and Hyperdrive experiment runs, a best performing model was selected for deployment.Finally we deploy model as a webservice for online real-time scoring. Request is sent to webservice to test deployed model. Below diagrams shows overview in pictorial form.

![Training architecuture snapshot](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/IntrusionDetectionSystemTrainingPart.png)

![Deployment architecuture snapshot](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/IntrusionDetectionSystemDeploymentPart.png)

In this I will showcase how we can use data science as a investigation tool for network traffic, here we use classfication algorithm to distinguish between normal traffic (good connections) and intrusion or attacks traffic (bad connections). A connection is a sequence of TCP packets starting and ending at some well difined times, between which data flows to and from source IP address  to a target IP address under some well defined protocol. We will create Intrusion Detection System (IDS). 

`Credits`:Ideas and references are taken from book Machine Learning and Security by Clarence Chio and David Freeman Published by O’Reilly Media, Inc., 1005 Gravenstein Highway North, Sebastopol, CA 95472.


## Project Set Up and Installation

If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, you should have Azure subscription. An Azure subscription allows you to manage storage, compute, and other assets in the Azure cloud. Following software and subscriptions are required to run note book.

  * Azure subscription
  * Create Azure ML workspace. An Azure ML workspace is an Azure resource that organizes and coordinates the action of many other Azure resources
    to assis in executing and sharing machine learning workflows.
  * Install Azure ML SDK, python 3.6, Pandas, numpy, and scikit librarires.
  * In this project we use Azure Container Instance (ACI) for deployment. An Azure subscription needs to be registered to use ACI.
  
 Overview of Azure environment snapshot is shown below
 
 ![Overview Azure enviroment](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/AzureOverview.png)

By fulfilling above requirements we can able to build Intrusion Detection System (IDS). IDS system is build using hyper drive concepts using Random Forest model (NetworkdataClassifier.py) implemented in hyperparameter_tuning.ipynb. Alternatively IDS model system is built using AutoML feature implemented in automl.ipynb. Best model is selected and deployed as webservice in Azure Container Instance (ACI).

## Dataset

### Overview
The dataset we use for Intrusion Detection System (IDS) is NSLKDD dataset, (A Detailed Analysis of the KDD CUP 99 Data Set,” Proceedings of the 2nd IEEE Symposium
on Computational Intelligence in Security and Defense Applications (2009): 53–58.) which is an improvement  to a classic network intrusion detection dataset used widely by security data science professionals. The original 1999 KDD Cup dataset was created for the DARPA Intrusion Detection Evaluation Program, prepared and managed by MIT Lincoln Laboratory. The data was collected over nine weeks and consists of raw tcpdump traffic in a local area network (LAN) that simulates the environment of a typical United States Air Force LAN. Some network attacks were deliberately carried out during the recording period. (Note: Description of data set is well described at https://www.unb.ca/cic/datasets/nsl.html)


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
The dataset was downloaded from  https://www.kaggle.com/hassan06/nslkdd to my desktop and then uploaded to my Github Repo. From my Github Repo I loaded to AML work space using "TabularDatasetFactory" class function "from_delimited_files" to create a TabularDataset to represent tabular data in .CSV delimited file using AML Azure SDK. 

Once Tabular data set was downloaded, it is converted to pandas data frame, cleaned it for accepting input for RandomForestClassifier by using API's get_dummies for categorial variables and Scalar object for continious variables. Here cleaning step step consists of data exploration, data cleaning, preprocessing, feature engineering. Here I checked data types (continious) for continious I used scaling features and created scalar object and stored for later to use it for testing data,  ordinal (categories) I used 'get_dummies'. This step is performed for Hyperdrive config model creation project.

In case AutomML project above clean up of data is not performed as it handled automatically by Azure AutoML module.In case of AutoML I uploaded csv file to default data store in Azure. After that I get data set pointer to data store for network training data and passed as an argument for "training_data" in AutoMLConfig API.

![Overview Azure enviroment](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/hyperdrive/DataSnapshotImportedToAzure.png)

Once the dataset was downloaded, SDK was again used to clean and split the data into training and validation datasets, which were then stored as Pandas dataframes in memory to facilitate quick data exploration and query, and registered as AML TabularDatasets in the workspace to enable remote access by the AutoML experiment running on a remote compute cluster.

## Automated ML

Intrusion Detection system is a classification task. According AutomMLConfigParameters are set which are used for this job as mentioned here.

AutoML Settings:

`max_cores_per_iteration` = 1 <br>
`max_concurrent_iterations` = 1 <br>
`featurization` = auto (by default so not set here in above) AutoMLConfig provides featurization arguments by default auto which provided learning features automatically.<br>
`n_cross_validations` = 5 Number of cross validations to perform configured are 5. <br>
`experiment timeout_minutes` = 30 minutes is set according to lab time provided.<br>
`primary_metric`= accuracy is used as provided dataset is balanced and is best suited for job at hand. <br>

AutoML Config:

`training_data`= Registered tabular data pointer in default data store is provided here. <br>
`blocked_modles` =  In this project I blocked XGBoostClassifier as I am facing issues in importing the model created in AutoML environment to compute envirnoment due to version
                    difference of XGBoost library. <br>
`label_column_name`= attack_type is one we have to predict if traffic is normal or attack. <br>
`compute_target` = The Azure Machine Learning compute target to run the Automated Machine Learning experiment on. <br>

AutoML is run on compute cluster named "cpu_cluster".  AutoML Run details snap shot is attached below

![AutomML Widget snapshot](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/automl/AutoMLWidgetOuput.png)

AutoML Best model with RunID

![AutomML best model with RunID](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/automl/AutoMLBestModelWithRunId.png)


### Results

Various model explored by AutomML snapshot is shown below

![AutomML various models explored](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/automl/AutoMLModelsExplored.png)

Auto ML gave best model accuracy of `0.9990`. Model selected is `StackEnsembleClassifier`and parameters of model is shown in below snapshot.

![AutomML best model parameters](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/automl/AutomMLBestModelDetails.png)

Confusion matrix of best model selected is shown below

![AutomML best model confusion matrix](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/automl/AutoMLConfusionMatrix.png)

## Hyperparameter Tuning

`DecisionTreeClassifier` is used for binary classification for network data traffic. Data reading, cleaning, transformation and model training is performed by me 
using pandas and scikit in `NetworkdataClassifier.py`.

`NetworkdataClassifier.py` accepts first argument as `-- criterion`(which measures quality of split), supported split criteria are “gini” for the Gini impurity and “entropy”
 for the information gain.

`NetworkdataClassifier.py` accepts second argument as `--max_depth`(the maximum depth of the tree). max_depth are sampled using choice(60,90, 120). Small max depth
 values correspond to small size trees and hight values of max depth corresponds to large trees.

Decision tree classifier is selected as it can handle categorical and real-valued features with ease—little to no preprocessing required. In this case we have lot of features and it is not clear from data,is data is linearly seperable so decision tree classifier is selected.

The HyperDrive experiment run was configured with parameter settings as follows:

Grid parameter sampling is chosen values in hyper parameters value space. In this scenario we don't have large search space and we have discrete parameter sampling. Grid sampling does a simple grid search over all possible values.

By specifying early termination policy we can automatically terminate poorly performing runs. Early termination improves computational efficiency. Bandit early termination policy is used to stop training if performance of current run is not with in the best run limits to avoid resource usage. Median stopping is an early termination policy based on running averages of primary metrics reported by the runs. This policy computes running averages across all training runs and terminates runs with primary metric values worse than the median of averages. I have choosen Bandit early for aggressive termination, where as median stopping can be used if we don't want aggresive termination.


### Results

Hyperdrive ML gave best model accuracy of `0.9987`. Decision tree classifier with parameters ['--criterion', 'gini', '--max_depth', '90'] is selected.
Hyperdrive `RunDetails` widget snapshot is shown below

![Hyperparam run details](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/hyperdrive/HyperdriveRunWidgetSnapshot.png)

![Hyperdrive runs](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/hyperdrive/HyperdriveChildRuns.png)

Best run hyperdrive decision tree classifier hyper parameters snapshot is provided below

![Hyper drive bestrun hyper parameters](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/hyperdrive/HyperdriveBestRunIDWithParameters.png)



## Model Deployment

Deployment is about delivering a trained model in to production so that it can be consumed by other. Here we deploy best model which is from AutoML with accuracy `0.9990`.
Here we deploy a model as a webservice on Azure Container Instance (ACI). Request is sent to service through scoring uri. Here we write entry script (`inference\autoscore.py`)
receives data submitted to a deployed web service, perform any actions that requires adaptation for inputs to model and `model.predict` is called. It then takes the response returned by the model and returns that to the client. The script is specific to your model. It must understand the data that the model expects and returns.

Below show input accept by IDS service.

![Service input and response](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/deploy/InputParametersAndResponse.png)

Below snap shot shows deployed end point in azure in healthy state.

![Deployed end point snapshot](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/deploy/DeployStatus.png)



## Screen Recording

Screen cast link

![screen cast link](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/CapstoneProjScreencast.mp4)

## Standout Suggestions
I have performed deployment of hyper model for study purpose how to use various standarization techniques developed during training to be used and applied for request. I studied about enabling application insights after deployment.

## Future work
Planning to work on following activities

* I am planning to work on indentifying type of attack if attack is detected. <br>
* Planning to work on conversion of registered model to ONNX format, allowing one to interchange models between various ML frameworks and tools.

## Training cluster deleted
Following snap shot shows training cluster is deleted after training.
![training cluster deleted](https://github.com/venkataravikumaralladi/AzureMLCapstoneProject/blob/main/snapshots/Deletesnapshot.png)
