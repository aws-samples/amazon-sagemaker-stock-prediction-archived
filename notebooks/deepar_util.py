
######################################################################
# This file contains utility functions to load test data from file,  #
# and invoke DeepAR predictor and plot the observed and target data. #
######################################################################

import io
import math
import json
import s3fs
import boto3
import datetime
import pandas as pd
import numpy as np
import sagemaker
import matplotlib
import matplotlib.pyplot as plt

# Function to Format DBG stock market data into a format suitable for DeepAR algorithm
def deeparize(stockdata, stocksymbols, interval, metrices = None):
    data_feed = pd.DataFrame()
    data_feed['CalcDateTime'] = pd.to_datetime(pd.Series(sorted(list(stockdata.CalcDateTime.unique()))),infer_datetime_format=True)
    data_feed.index = data_feed['CalcDateTime']
    data_feed.drop('CalcDateTime', axis=1, inplace = True)

    for mnemonic in stocksymbols:

        mnemonic_data = stockdata[stockdata.Mnemonic == mnemonic].copy()
        mnemonic_data.index = mnemonic_data['CalcDateTime']
        mnemonic_data = mnemonic_data.sort_index()
        mnemonic_data = mnemonic_data.iloc[:,-6:]
        if metrices is None:
            metrices = mnemonic_data.columns.values
        for col in metrices:
            metric_col = mnemonic_data[col].to_frame()
            metric_col.columns = ["{}-{}".format(mnemonic,col)]
            data_feed = data_feed.add(metric_col, fill_value=0)            
            
    data_feed = data_feed.resample(interval).mean()       
    data_feed.fillna(method='backfill', limit=1, inplace=True)
    data_feed.fillna(method='ffill', inplace=True) 
    data_feed.fillna(value=0, inplace=True)
    return data_feed


# Function to load resampled stock data from a specified S3 location
def load_resampled_from_s3(interval, bucket, s3_data_key, mnemonics=None, metrices = None):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key="{}/{}/resampled_stockdata.csv".format(s3_data_key, interval))
    loaded = pd.read_csv(io.BytesIO(obj['Body'].read()), parse_dates=True)
    if mnemonics is None:
        mnemonics = list(loaded.Mnemonic.unique())
    return deeparize(loaded, mnemonics, interval, metrices), mnemonics


# Function to plot specified metrices for specified stock, each separate plot
def metrics_plot(mnemonics, metrics = None, data=None, interval = None, bucket = None, s3_key = None):
    
    if data is None and interval is not None and bucket is not None and s3_key is not None:
        data, symbols = load_resampled_from_s3(interval, bucket, s3_key)  
        
    if metrics is None:
        metrics = ['MinPrice', 'MaxPrice', 'StartPrice', 'EndPrice', 'TradedVolume', 'NumberOfTrades']

    fig, axs = plt.subplots(math.ceil((len(metrics) * len(mnemonics))/3), 3, figsize=(20, 20), sharex=True)
    axx = axs.ravel()
    i = 0
    for mnemonic in mnemonics:
        for metric in metrics:
            data["{}-{}".format(mnemonic,metric)].head()
            data["{}-{}".format(mnemonic,metric)].plot(ax=axx[i])
            axx[i].set_xlabel("date")    
            axx[i].set_ylabel("{}-{}".format(mnemonic,metric))   
            axx[i].grid(which='minor', axis='x')
            axx[i].set_xticklabels(data.index, rotation=90) 
            i = i+1
            
            
# Function to plot specified metrices for specified stock, all superimposed on a single plot            
matplotlib.rcParams['figure.figsize'] = (25, 17) # use bigger graphs
def timeseries_plot(mnemonics, metrics, data=None, interval = None, bucket = None, s3_key = None):

    if data is None and interval is not None and bucket is not None and s3_key is not None:
        data, symbols = load_resampled_from_s3(interval, bucket, s3_key)      
    ax = None
    for mnemonic in mnemonics:
        selected = pd.DataFrame()
        selected['CalcDateTime'] = pd.Series(sorted(list(data.index.unique())))
        selected.index = selected['CalcDateTime']
        selected = selected.sort_index()
        selected.drop('CalcDateTime', axis=1, inplace = True)
        for metric in metrics:
            selected[metric] = data["{}-{}".format(mnemonic,metric)]
        selected_columns = list(selected.columns)
        for i, column in enumerate(selected_columns):
            selected_columns[i] = "{}-{}".format(mnemonic, column)
        selected.columns = selected_columns          
        ax = selected.plot( ax = ax)  
        ax.set_xticklabels(data.index, rotation=90) 
        
        
# Function to convert data frames containing time series data to JSON serialized data that DeepAR works with
def json_serialize(data, start, end, target_column, covariate_columns, interval):
    timeseries = {}

    for i, col in enumerate(data.columns):
        metric = col[col.find('-')+1:]
        stock = col[:col.find('-')]
        if metric == target_column:
            if stock in timeseries.keys():
                timeseries[stock]["target"] = data.iloc[:,i][start:end]
            else:
                timeseries[stock] = {}
                timeseries[stock]["start"] = str(pd.Timestamp(datetime.datetime.strptime(str(start), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S"), freq = interval))
                timeseries[stock]["target"] = data.iloc[:,i][start:end]            
            print("Time series for {} added".format(stock))
        elif metric in covariate_columns:
            if stock in timeseries.keys():
                if "dynamic_feat" in timeseries[stock]:
                    dynamic_feat = timeseries[stock]["dynamic_feat"]
                    dynamic_feat.append(data.iloc[:,i][start:end])
                else:
                    dynamic_feat = []
                    dynamic_feat.append(data.iloc[:,i][start:end])
                    timeseries[stock]["dynamic_feat"] = dynamic_feat
            else:
                timeseries[stock] = {}
                dynamic_feat = []
                dynamic_feat.append(data.iloc[:,i])
                timeseries[stock]["dynamic_feat"] = dynamic_feat            
            print("Dynamic Feature - {} for {} added".format(metric, stock))
        else:
            pass

    json_data = [
        {
            "start": ts["start"],
            "target": ts["target"].tolist(),  
            "dynamic_feat": [feat.tolist() for feat in ts["dynamic_feat"]]
        }
        for ts in timeseries.values()
    ]   
    return json_data


# Function to first split the data into training and test sets, and then to JSON serialize both sets
def generate_train_test_set(data, target_column, covariate_columns, interval, train_test_split=0.9, num_test_windows=4):
    num_samples = len(data.index.values)
    num_train = int(train_test_split * num_samples)
    num_test = int((num_samples - num_train)/num_test_windows)
    print("Sample Size = {}, Training Set: {}, Test Set: {} * {}".format(num_samples, num_train, num_test_windows, num_test))
    train_start_dt = data.index[0]
    train_end_dt = data.index[num_train - 1]   
    print("Training Set: Starts at - {}, Ends at - {}".format(train_start_dt, train_end_dt))
    train_data = json_serialize(data, train_start_dt, train_end_dt, target_column, covariate_columns, interval)
    test_data = []
    test_start_date = train_start_dt
    for i in range(num_test_windows):
        test_end_dt = data.index.values[num_train + i*num_test - 1]
        test_data.extend(json_serialize(data, test_start_date, test_end_dt, target_column, covariate_columns, interval))
    return train_data, test_data, train_start_dt, train_end_dt


#Function to write JSON serialized training and test data into S3 bucket, which will later be fed to training container
def write_dicts_to_file(data, interval, bucket, path, channel):
    fs = s3fs.S3FileSystem()
    with fs.open("{}/{}/{}/{}/{}.json".format(bucket, path, interval, channel, channel), 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))
    return "s3://{}/{}/{}/{}/".format(bucket, path, interval, channel)        


# Class that allows making requests using pandas Series objects rather than raw JSON strings
class DeepARPredictor(sagemaker.predictor.RealTimePredictor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, content_type=sagemaker.content_types.CONTENT_TYPE_JSON, **kwargs)
        
    def predict(self, ts, cat=None, dynamic_feat=None, 
                num_samples=100, return_samples=False, quantiles=["0.1", "0.5", "0.9"]):
        """Requests the prediction of for the time series listed in `ts`, each with the (optional)
        corresponding category listed in `cat`.
        
        ts -- `pandas.Series` object, the time series to predict
        cat -- integer, the group associated to the time series (default: None)
        num_samples -- integer, number of samples to compute at prediction time (default: 100)
        return_samples -- boolean indicating whether to include samples in the response (default: False)
        quantiles -- list of strings specifying the quantiles to compute (default: ["0.1", "0.5", "0.9"])
        
        Return value: list of `pandas.DataFrame` objects, each containing the predictions
        """
        prediction_time = ts.index[-1] + 1
        quantiles = [str(q) for q in quantiles]
        req = self.__encode_request(ts, cat, dynamic_feat, num_samples, return_samples, quantiles)
        res = super(DeepARPredictor, self).predict(req)
        return self.__decode_response(res, ts.index.freq, prediction_time, return_samples)
    
    def __encode_request(self, ts, cat, dynamic_feat, num_samples, return_samples, quantiles):
        instance = series_to_dict(ts, cat if cat is not None else None, dynamic_feat if dynamic_feat else None)

        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles", "samples"] if return_samples else ["quantiles"],
            "quantiles": quantiles
        }
        
        http_request_data = {
            "instances": [instance],
            "configuration": configuration
        }
        
        return json.dumps(http_request_data).encode('utf-8')
    
    def __decode_response(self, response, freq, prediction_time, return_samples):
        # we only sent one time series so we only receive one in return
        # however, if possible one will pass multiple time series as predictions will then be faster
        predictions = json.loads(response.decode('utf-8'))['predictions'][0]
        prediction_length = len(next(iter(predictions['quantiles'].values())))
        prediction_index = pd.DatetimeIndex(start=prediction_time, freq=freq, periods=prediction_length)        
        if return_samples:
            dict_of_samples = {'sample_' + str(i): s for i, s in enumerate(predictions['samples'])}
        else:
            dict_of_samples = {}
        return pd.DataFrame(data={**predictions['quantiles'], **dict_of_samples}, index=prediction_index)

    def set_frequency(self, freq):
        self.freq = freq
        
def encode_target(ts):
    return [x if np.isfinite(x) else "NaN" for x in ts]        


def series_to_dict(ts, cat=None, dynamic_feat=None):
    """Given a pandas.Series object, returns a dictionary encoding the time series.

    ts -- a pands.Series object with the target time series
    cat -- an integer indicating the time series category

    Return value: a dictionary
    """
    obj = {"start": str(ts.index[0]), "target": encode_target(ts)}
    if cat is not None:
        obj["cat"] = cat
    if dynamic_feat is not None:
        obj["dynamic_feat"] = dynamic_feat        
    return obj

# Function to create a data structure to invoke prediction for a given stock and within a given time range
def query_for_stock(stock_to_predict, target_column, covariate_columns, data, prediction_length, start = None, end = None):
    if start is None:
        start = data.index.values[0]
    if end is None:
        end = data.index.values[-1]
    startloc = data.index.get_loc(start)
    endloc = data.index.get_loc(end)
    stockts = None
    ts = None
    dynamic_feat = []
    
    for i, col in enumerate(data.columns):
        stock = col[:col.find('-')]
        metric = col[col.find('-')+1:]
        if stock == stock_to_predict: 
            if metric == target_column:
                ts = data.iloc[:,i][startloc:endloc-prediction_length]
                stockts = data.iloc[:,i][:]
                print("Time series - {} for {} selected".format(metric, stock))
            elif metric in covariate_columns:
                dynamic_feat.append(data.iloc[:,i][startloc:endloc].tolist())
                print("Dynamic Feature - {} for {} selected".format(metric, stock))
            else:
                pass
    return ts, dynamic_feat, stockts

def plot_predicted_observed_at_quantile(ts, observed, predicted, quantile):
    ax = None
    ax = observed.plot( ax = ax )
    ax.set_xticklabels(observed.index, rotation=90)
    #for col in prediction.columns:
    predicted = ts.append(predicted[quantile])
    predicted.plot(ax = ax)


def plot(
    predictor, 
    stock_id,
    mnemonics,
    target_ts, 
    target_column,
    covariate_columns,
    prediction_length,
    plot_history,    
    cat=None, 
    dynamic_feat=None, 
    forecast_date=None, 
    show_samples=False, 
    confidence=75
):
    if forecast_date is None:
        forecast_date = target_ts.index[-1]
    print("calling served model to generate predictions starting from {}".format(str(forecast_date)))
    assert(confidence > 50 and confidence < 100)
    low_quantile = 0.5 - confidence * 0.005
    up_quantile = confidence * 0.005 + 0.5
    
    ts, dynamic_feat, stockts = query_for_stock(mnemonics[stock_id], target_column, covariate_columns, target_ts, prediction_length, end=forecast_date)
    args = {
        "ts": ts,
        "return_samples": show_samples,
        "quantiles": [low_quantile, 0.5, up_quantile],
        "num_samples": 100
    }

    if dynamic_feat is not None:
        args["dynamic_feat"] = dynamic_feat
        fig = plt.figure(figsize=(20, 6))
        ax = plt.subplot(2, 1, 1)
    else:
        fig = plt.figure(figsize=(20, 3))
        ax = plt.subplot(1,1,1)
    
    if cat is not None:
        args["cat"] = cat
        ax.text(0.9, 0.9, 'cat = {}'.format(cat), transform=ax.transAxes)

    # call the end point to get the prediction
    prediction = predictor.predict(**args)
    # plot the samples
    if show_samples: 
        for key in prediction.keys():
            if "sample" in key:
                prediction[key].plot(color='lightskyblue', alpha=0.2, label='_nolegend_')
                
                
    # plot the target
    target_section = stockts[forecast_date-plot_history:forecast_date+prediction_length]
    target_section.plot(color="black", label='target')
    
    # plot the confidence interval and the median predicted
    ax.fill_between(
        prediction[str(low_quantile)].index, 
        prediction[str(low_quantile)].values, 
        prediction[str(up_quantile)].values, 
        color="b", alpha=0.3, label='{}% confidence interval'.format(confidence)
    )
    prediction["0.5"].plot(color="b", label='P50')
    ax.legend(loc=2)    
    
    # fix the scale as the samples may change it
    #ax.set_ylim(target_section.min() * 0.5, target_section.max() * 1.5)
    ax.set_ylim(ts.min(), ts.max())
    
    '''
    if dynamic_feat is not None:
        for i, f in enumerate(dynamic_feat, start=1):
            ax = plt.subplot(len(dynamic_feat) * 2, 1, len(dynamic_feat) + i, sharex=ax)
            feat_ts = pd.Series(
                index=pd.DatetimeIndex(start=target_ts.index[0], freq=target_ts.index.freq, periods=len(f)),
                data=f
            )
            feat_ts[forecast_date-plot_history:forecast_date+prediction_length].plot(ax=ax, color='g')
    '''