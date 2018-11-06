###########################################################
# This file contains utility functions to load test data  #
# from file, and invoke custom rnn predictor either using #
# file location on S3, or serialized CSV data.            #
###########################################################

import io
import re
import json
import boto3
import tarfile
import pandas as pd
from matplotlib import pyplot
import matplotlib as mpl
import matplotlib.pyplot as plt

# Since hyperparameters were saved into a JSON file in the same location as model output during training,
# it is present inside the model archive.
# This function extracts the JSON file from within the archive and loads into a Python dictionary
# This may be used to obtain the parameters used during training, if needed, while invoking prediction endpoint.
def load_json_from_s3_tarfile(s3, bucket, archive, filename):
    
    bytestream = io.BytesIO(s3.get_object(Bucket=bucket, Key=archive)['Body'].read())
    tarf = tarfile.open(fileobj=bytestream)
    paramsfile = tarf.extractfile(filename)
    params = json.load(paramsfile)
    
    return params


# Since during training test data were split and saved into a CSV file in the same location as model output,
# it is present inside the model archive.
# This function extracts the specified CSV file from within the archive and loads into a data frame.
def load_csv_from_s3_tarfile(s3, bucket, archive, filename, outputpath = None, index = None, sep=","):
    bytestream = io.BytesIO(s3.get_object(Bucket=bucket, Key=archive)['Body'].read())
    compressesfile= tarfile.open(fileobj=bytestream)
    extractedfile = compressesfile.extractfile(filename)  
    
    if index is not None and type(index) is int:
        data = pd.read_csv(extractedfile, sep, index_col=index)
    else:
        data = pd.read_csv(extractedfile, sep)

    if outputpath is not None and type(outputpath) == str:
        outputprefix = "{}/{}".format(outputpath, filename)
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, outputprefix).put(Body=csv_buffer.getvalue())        
        
    return data


# Since during training, we created several test samples and saved into CSV files in the same location as model output,
# it is present inside the model archive.
# This function extracts the archive and lists CSV files, matching name pattern, that contains such test samples.
def extract_matching_csv_files_from_s3_tarfile(s3, bucket, archive, namepattern, outputpath = None, index = None, sep=",", parse_dates = True):
    
    bytestream = io.BytesIO(s3.get_object(Bucket=bucket, Key=archive)['Body'].read())
    filenames = list(filter( lambda s: re.compile(namepattern).match(s), tarfile.open(fileobj=bytestream).getnames()))
    filepaths = []
    
    for filename in filenames:
        
        bytestream = io.BytesIO(s3.get_object(Bucket=bucket, Key=archive)['Body'].read())
        tarf = tarfile.open(fileobj=bytestream)
        extractedfile = tarf.extractfile(filename)
        
        if index is not None and type(index) is int:
            data = pd.read_csv(extractedfile,index_col=index, delimiter=sep,parse_dates=parse_dates)
        else:
            data = pd.read_csv(extractedfile,delimiter=sep,parse_dates=parse_dates)

        
        if outputpath is not None and type(outputpath) == str:
            outputprefix = "{}/{}".format(outputpath, filename)
            output_s3_path = "s3://{}/{}".format(bucket, outputprefix)
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer)
            s3_resource = boto3.resource('s3')
            s3_resource.Object(bucket, outputprefix).put(Body=csv_buffer.getvalue())   
            filepaths.append(output_s3_path)
        else:
            filepaths.append(filename)
            
    return filepaths


# Given a data frame containing test data, this function splits it into multiple samples,
# at specified increments of date, and returns a series of data frames.
# This function is useful for generating several predictions using data at certain intervals
# from within the whole test set.
def test_sample_from_testdata(testdata, target_stock, covariate_stocks, lag, horizon, inc):
    
    target_stock_data = testdata[testdata.Mnemonic == target_stock].copy()
    covariate_stock_data = []
    covariate_stock_list = covariate_stocks.split(",")
    
    for covariate_stock in covariate_stock_list:
        covariate_stock_data.append(testdata[testdata.Mnemonic == covariate_stock.strip()])
        
    test_size = target_stock_data.shape[0] 
    span = lag + horizon + 1 
    testinputs = []
    num_test_samples = int(test_size/span)
    
    for i in range(0, test_size - span, inc):  
        
        test_input = target_stock_data.iloc[i:i+span]
        for cov in covariate_stock_data:
            test_input = test_input.append(cov.iloc[i:i+span])
        testinputs.append(test_input)  
        
    return testinputs    


# Given a data frame containing test data, and an index position, this function generates a single prediction input
# spanning backward equal to specified lag, and forward equal to specified horizon
# This is useful when you want to just generate one prediction on a given date.
def test_input_for_date(testdata, forecast_date_index, target_stock, covariate_stocks, lag, horizon):
    
    target_stock_data = testdata[testdata.Mnemonic == target_stock].copy()
    covariate_stock_data = []
    covariate_stock_list = covariate_stocks.split(",")
    
    for covariate_stock in covariate_stock_list:
        covariate_stock_data.append(testdata[testdata.Mnemonic == covariate_stock.strip()])
        
        
    test_input = target_stock_data.iloc[forecast_date_index - lag:forecast_date_index+horizon+1]
    for cov in covariate_stock_data:
        test_input = test_input.append(cov.iloc[forecast_date_index - lag:forecast_date_index+horizon+1])
        
    return test_input  


# This function is used to plot the loss history, from the CSV file containing the netwrok loss after each epoch,
# as saved during the training, at the same location as model, therefore available within the model archive.
# This plot is useful to ascertain that training loss is asymptotically decreasing as training progresses.
def plot_loss(s3, bucket, archive, filename):

    mpl.rcParams['figure.figsize'] = (15, 10) # use bigger graphs

    # plot history
    lossdata  = load_csv_from_s3_tarfile(s3, bucket, archive, filename)
    pyplot.plot(lossdata.combo_out_loss, label='Combo Loss')
    pyplot.plot(lossdata.main_out_loss, label='Main Loss')
    pyplot.plot(lossdata.loss, label='Loss')
    pyplot.legend()
    pyplot.show()

    
# This function, given a list of CSV files on S3 containing test data, and a predictor,
# invokes the predictor endpoint with S3 path, ontains the results and
# plots the predicted series of values, alongwith the test data series,
# thus providing a way to visually compare the prediction outcome for each test set
def plot_sample_predictions(predictor, filepaths, target_stock, target_column, lag):
    
    fig, axs = plt.subplots(int((len(filepaths)-1)/3)+1, 3, figsize=(30, 10))
    axx = axs.ravel()  

    for k, filepath in enumerate(filepaths):
        
        data = pd.read_csv(filepath , index_col = 0)
        test_main = data[data.Mnemonic == target_stock].copy()  
        given = test_main[target_column].iloc[0:-1]
        given.plot(ax = axx[k], use_index=True)
        preds = predictor.predict(filepath).decode("utf-8").split()
        
        for i, pred in enumerate(preds):
            preds[i] = float(pred[pred.find(',')+1:])
            
        predicted = test_main[target_column].iloc[:lag].append(pd.Series(preds))
        predicted.index = given.index
        predicted = predicted.reset_index()
        axx[k].set_xticklabels(predicted['CalcDateTime'], rotation=90)
        predicted[0].plot(ax = axx[k])
        
        
# This function, given a data frame containing a set of data, and a predictor,
# first generates a set of test inputs at specified increment, loads and serializes data for each interval
# and then invokes the predictor with serialized CSV and plots the predicted series of values, alongwith the test data series.
# While plotting the outcome this function generates separate plots just showing observed vs predicted values
# thus providing a way to visually compare the prediction outcome for each test set        
def plot_sample_test_performance(predictor, testdata, target_stock, covariate_stock, target_column, lag, horizon, inc):

    testinputs = test_sample_from_testdata(testdata, target_stock, covariate_stock, lag, horizon, inc)

    fig, axs = plt.subplots(int((len(testinputs)-1)/3)+1, 3, figsize=(30, 30))
    axx = axs.ravel()  
    
    for k, testinput in enumerate(testinputs):
        test_main = testinput[testinput.Mnemonic == target_stock].copy()  
        given = test_main[target_column].iloc[0:-1]
        given.plot(ax = axx[k], use_index=True)
        prediction = predictor.predict(testinput.to_csv())
        preds = prediction.decode("utf-8").split()
        
        for i, pred in enumerate(preds):
            preds[i] = float(pred[pred.find(',')+1:])
            
        predicted = test_main[target_column].iloc[:lag].append(pd.Series(preds))
        predicted.index = given.index
        predicted = predicted.reset_index()
        axx[k].set_xticklabels(predicted['CalcDateTime'], rotation=90)
        predicted[0].plot(ax = axx[k])                
        
        
# This function, given a data frame containing a set of data, and a predictor,
# first generates a set of test inputs at specified increment, loads and serializes data for each interval
# and then invokes the predictor with serialized CSV and plots the predicted series of values, alongwith the test data series.
# While plotting the outcome, this function plots the overall test set, and super imposes prediction outcome 
# for each sample sets thus providing a way to visually compare the prediction outcome for each test set         
def plot_overall_test_performance(predictor, testdata, target_stock, covariate_stock, target_column, lag, horizon, inc):

    testinputs = test_sample_from_testdata(testdata, target_stock, covariate_stock, lag, horizon, inc)
        
    mpl.rcParams['figure.figsize'] = (25, 17) 
    ax = None
    target_stock_data = testdata[testdata.Mnemonic == target_stock].copy()
    
    given = target_stock_data[target_column]
    ax = given.plot( ax = ax )
    ax.set_xticklabels(target_stock_data.index, rotation=90)

    for k, testinput in enumerate(testinputs):
        prediction = predictor.predict(testinput.to_csv())
        preds = prediction.decode("utf-8").split()
        
        for i, pred in enumerate(preds):
            preds[i] = float(pred[pred.find(',')+1:])
            
        predicted = target_stock_data[target_column].iloc[:k*inc+1+lag].append(pd.Series(preds))  
        predicted = predicted.reset_index()
        predicted[0].plot(ax = ax)   
        
        
# Given a data frame containing a set of data, this function generates a series of date values,
# based on the datetime index of the data frame, which can be used to specify the range of values
# in an ipython selection widget.
def get_date_range(data, target_stock, dateformat, lag, horizon, interval):
    
    target_data = data[data.Mnemonic == target_stock].copy()
    dateoptions = [(date.strftime(dateformat), date) 
                   for date in pd.date_range(target_data.index[lag], target_data.index[-(horizon+1)], freq=interval)]
    
    return dateoptions


# Given a data frame containing a set of data, a predictor and an index position within the data set,
# fthis function first generates a input set at specified position, loads and serializes data 
# and then invokes the predictor with serialized CSV and plots the predicted series of values, alongwith the test data series.
# While plotting the outcome, this function plots the overall test set, and super imposes prediction outcome 
# for each sample sets thus providing a way to visually compare the prediction outcome for each test set  
def predict_and_plot(predictor, testdata, forecast_date_index, target_stock, covariate_stocks, target_column, lag, horizon):
    
    testinput = test_input_for_date(testdata, forecast_date_index, target_stock, covariate_stocks, lag, horizon)
    
    mpl.rcParams['figure.figsize'] = (25, 17) 
    ax = None
    target_stock_data = testdata[testdata.Mnemonic == target_stock].copy()    
    
    given = target_stock_data[target_column]
    ax = given.plot( ax = ax )
    ax.set_xticklabels(target_stock_data.index, rotation=90)    
    
    prediction = predictor.predict(testinput.to_csv())
    preds = prediction.decode("utf-8").split()

    for i, pred in enumerate(preds):
        preds[i] = float(pred[pred.find(',')+1:])   
        
    predicted = target_stock_data[target_column].iloc[:forecast_date_index].append(pd.Series(preds))  
    predicted = predicted.reset_index()
    predicted[0].plot(ax = ax)           
    