# This is the file that implements a flask server to do inferences. 
# It's the file that you will modify to implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import io
import sys
import signal
import traceback
import flask
import pdb
import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing
from keras.models import load_model
import keras
import tensorflow as tf

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
output_path = os.path.join(prefix, 'output')
saved_param_path = os.path.join(model_path, 'hyperparameters.json')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            model_artifact = os.path.join(model_path, 'rnn-combo-model.h5')
            if os.path.exists(model_artifact):
                cls.model = load_model(model_artifact)                
                cls.model._make_predict_function()
                cls.graph = tf.get_default_graph()    
            else:
                print("Model not found.")                
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        return model.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

# Function to normalize ML inputs
def normalize_input(df, traindata):
    data = pd.concat([traindata,df],axis=0)
    data = data.diff() 
    data = data.replace(np.nan, 0)
    scaler = preprocessing.StandardScaler() 
    for feat in data.columns:
        transformed_feature = scaler.fit_transform(data.eval(feat).values.reshape(-1,1))
        data[feat] = transformed_feature
    return data

# Function to denormalize ML outputs
def denormalize_output(array, traindata):
    traindata = traindata.diff()
    
    traindata = traindata.replace(np.nan, 0) # Doesn't work if traindata is not a df 
    scaler = preprocessing.StandardScaler() # or: MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(traindata['TargetMetric'].values.reshape(-1,1)) 

    new = scaler.inverse_transform(array.reshape(-1,1)) #df.values.reshape(-1,1))
    return new

# Function to load data for training.
# This function splits the data into training and test set according to the specified interval 
# and also creates sets of samples with number of observations equal to specified lag concatenated as X 
# and number of observations equal to specified horizon concatenated as Y
def load_data_for_prediction(stock, lag, horiz):  
    data = stock.as_matrix() 
    lags = []
    horizons = []

    nsample = len(data) - lag - horiz  # Number of time series to be predicted upon
    for i in range(nsample): 
        lags.append(data[i: i + lag , :]) 
        horizons.append(data[i + lag : i + lag + horiz, -1])

    lags = np.array(lags)
    horizons = np.array(horizons)
    
    x_test = lags
    y_test = horizons 

    return [x_test, y_test]

def date_part(dt):
    return str(dt).split(' ')[0]

# The predictor is designed to load data for prediction either from a specified S3 location or from serialized CSV

# This function reads CSV file stored in specified S3 location directly into a data frame
def data_from_s3_file(dataloc):
    print("Reading data from CSV file at {}".format(dataloc))
    bucketstart = dataloc.find('s3://')+5 if dataloc.find('s3://') >= 0 else 0
    bucketend = dataloc.find('/',bucketstart)
    bucket = dataloc[bucketstart:bucketend]
    key = dataloc[bucketend+1:]   
    data_filename = "s3://{}/{}".format(bucket, key)
    data = pd.read_csv(data_filename, index_col=0, parse_dates=True)
    return data

# This function loads a data frame from serialized string passed directly to the endpoint
def data_from_csv_string(csvstr):
    print("Reading data from CSV string input")
    csvlines = csvstr.split("\n")
    headers = []
    columns = {}
    for i, line in enumerate(csvlines):
        if len(line.strip()) > 0:
            if len(headers) <= 0:
                headers = line.strip().split(",")
            else:
                values = line.strip().split(",")
                for j, value in enumerate(values):
                    val = None
                    if headers[j].find("Date") >= 0:
                        if value.find(" ") > 0:
                            val = datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
                        else:
                            val = datetime.datetime.strptime(value, '%Y-%m-%d')                        
                    elif headers[j].find("Mnemonic") >= 0:
                        val = value
                    else:
                        val = float(value)

                    if headers[j] in columns.keys():
                        columns[headers[j]].append(val)
                    else:
                        col = []
                        col.append(val)
                        columns[headers[j]] = col
    data = pd.DataFrame.from_dict(columns)
    data = data.set_index(data.columns[0])  
    return data

@app.route('/ping', methods=['GET'])
def ping():
    #Determine if the container is working and healthy. In this sample container
    # Declare it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None 
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    #Do an inference on a single batch of data. 
    # Accept either a n S3 location containing a CSV file or from a serialized string, 
    # convert it to a pandas data frame for internal use and 
    # convert the predictions back to CSV 
     
    with open(saved_param_path, 'r') as tc:
        savedParams = json.load(tc)
    print("Hyperparameters file : " + json.dumps(savedParams))           
    interval = savedParams.get('interval', 'D').upper()
    lag=int(savedParams.get('lag', '10'))
    horizon=int(savedParams.get('horizon', '5')) 
    target_stock = savedParams.get('target_stock', 'BMW').upper()
    covariate_stocks = savedParams.get('covariate_stocks', 'CON, DAI, PAH3, VOW3').upper()
    covariates = covariate_stocks.split(',')
    for i, cov in enumerate(covariates):
        covariates[i] = cov.strip()        
    target_column = savedParams.get('target_column', 'EndPrice')
    covariate_columns = savedParams.get('covariate_columns', 'StartPrice, MinPrice, MaxPrice')
    covariate_columns = covariate_columns.split(',')
    for i, covcol in enumerate(covariate_columns):
        covariate_columns[i] = covcol.strip()  
            
    data = None

    # Load CSV to DataFrame
    if flask.request.content_type != 'text/csv':
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    csvinput = flask.request.data.decode('utf-8')
    if csvinput.find("/") >= 0:
        data = data_from_s3_file(csvinput)        
    else:
        data = data_from_csv_string(csvinput)

    # Read training data to set scaler parameters based on training data
    file = os.path.join(model_path, 'traindata.csv') 
    traindata = pd.read_csv(file, index_col = 0, parse_dates=True)
    print("Original training data read back, to be used for normalization")

    print('Rescaling {}'.format(target_stock))
    
    # Main time series
    test_main = data[data.Mnemonic == target_stock].copy()
    test_main['TargetMetric'] = test_main[target_column]
    test_main.drop(['Mnemonic', target_column], 1, inplace=True)    
    test_cols = test_main.columns.values
    for col in test_cols:
        if col != 'TargetMetric' and col not in covariate_columns:
            test_main.drop(col, 1, inplace=True)
    test_main_ori = test_main
    
    
    train_main = traindata[traindata.Mnemonic == target_stock].copy()    
    train_main['TargetMetric'] = train_main[target_column]
    train_main.drop(['Mnemonic', target_column], 1, inplace=True)    
    train_cols = train_main.columns.values
    for col in train_cols:
        if col != 'TargetMetric' and col not in covariate_columns:
            train_main.drop(col, 1, inplace=True)

    test_main = normalize_input(test_main,train_main)
    print("Main time series normalized : " + str(test_main.shape))


    # Exogenous time series
    test_exo = pd.DataFrame()
    test_exo['CalcDateTime'] = pd.to_datetime(pd.Series(sorted(list(data.index.unique()))),infer_datetime_format=True)
    test_exo.index = test_exo['CalcDateTime']
    test_exo.drop('CalcDateTime', axis=1, inplace = True)
    logoutput = []
    for n, covariate_stock in enumerate(covariates):
        if target_stock != covariate_stock:
            logoutput.append('Rescaling {}'.format(covariate_stock))
            if len(logoutput) == 7:
                print("\t".join(logoutput))
                logoutput = []
            exo = data[data.Mnemonic == covariate_stock].copy()
            
            exo['TargetMetric'] = exo[target_column]
            exo.drop(['Mnemonic', target_column], 1, inplace=True)
            test_cols = exo.columns.values
            for col in test_cols:
                if col != 'TargetMetric' and col not in covariate_columns:
                    exo.drop(col, 1, inplace=True)  

            train_exo = traindata[traindata.Mnemonic == covariate_stock].copy()
            
            train_exo['TargetMetric'] = train_exo[target_column]
            train_exo.drop(['Mnemonic', target_column], 1, inplace=True)
            train_cols = train_exo.columns.values
            for col in train_cols:
                if col != 'TargetMetric' and col not in covariate_columns:
                    train_exo.drop(col, 1, inplace=True)                       
            
            exo = normalize_input(exo,train_exo)

            #exo = normalize_data(exo)    
            exo = exo.sort_index()
            for col in exo.columns.values:
                metric_col = exo[col].to_frame()
                metric_col.columns = ["{}-{}".format(covariate_stock,col)]
                test_exo = test_exo.add(metric_col, fill_value=0)   
    print("\t".join(logoutput))
    test_exo.dropna(how='all', inplace=True)
    print("Exogenous time series normalized : " + str(test_exo.shape))  
    
    print("Starting inference")

    
    ########################################################### 
    #                                                         #
    #  Select input features and run model to make prediction # 
    #                                                         #
    ###########################################################

    Xmain_test, dummy = load_data_for_prediction(test_main, lag, horizon)
    print("Main time series loaded for prediction")
    Xexo_test, dummy = load_data_for_prediction(test_exo, lag, horizon)    
    print("Exogenous time series loaded for prediction")
        
    # Get the undifferenced data
    ori, dummy = load_data_for_prediction(test_main_ori, lag, horizon)

    # Do the prediction
    predictions = ScoringService.predict({'main_in': Xmain_test, 'exo_in': Xexo_test})    
    print("Prediction done by RNN Model")
    
    # Rescale back to original target range
    pred = denormalize_output(predictions[0][0],train_main) 
    print("Predicted values de-normalized")
    
    # Add predicted trends to actual values
    pred[0,0] = ori[0,-1,-1] + pred[0,0]
    for i in range(1,len(pred[:,0])):
        pred[i,0] = pred[i-1,0] + pred[i,0]      

    # Convert back to CSV
    result = ""
    for i in range(pred.shape[0]):
        result = result + str(i) + "," + str(pred[i,0])+"\n"
    print(result)

    print("Completed inference")

    return flask.Response(response=result, status=200, mimetype='text/csv')