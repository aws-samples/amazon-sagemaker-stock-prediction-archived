# Stock Prediction using Neural Networks on Amazon SageMaker

## Introduction

This is a sample workshop that demonstrates how to use neural networks based algorithm for time series prediction. The workshop uses stock market data maintained by Deutsche Börse under [Registry of open data](https://registry.opendata.aws/deutsche-boerse-pds/) on AWS. This dataset continaes minute by minute stock movement data from EU market, containing 100+ securities, tracked since July, 2016.

Time series data can be analysed using a variety of techniques, ranging from a simple Multi Layer Perceptron, to a stacked Recurrent Neural Network, using forecasting methods such as Autoregressive Integrated Moving Average (ARIMA) or Exponential Smoothing (ETS). As a first attempt, we'll use a simple RNN based model to predict stock price a single security.

## License Summary

This sample code is made available under a modified MIT license. See the LICENSE file.

## Action Plan

[Amazon SageMaker](https://aws.amazon.com/sagemaker/), is the Machine Learning platform on AWS that provides infrastructure to run hosted Jupyetr Notebooks. Being integrated with other storage and analytics services on AWS, data collection, preparation and visualization, all essential tasks for a successful Machine Learning project becomes more secured and streamlined on SageMaker. 

![SageMaker](./images/sagemaker.png)

In this workshop, we'll use SageMaker hosted notebooks to fetch the data from Deutsche Börse dataset, clean up and aggregate the data on [Amazon S3](https://aws.amazon.com/s3/) buckets. We'll also use [Amazon Athena](https://aws.amazon.com/athena/) to query the data and [Amazon QuickSight](https://aws.amazon.com/quicksight/) to visuaize the data. This will allows us to develop an intuition about the nature of the data.

In addition to hosted Notebooks, SageMaker also provdes managed training and hosting for Machine Learning models, using a variety of languages and libraries. In our first attempt, after we build a model locally, we'll use this functionality to containerize the training and prediction code, publish on an [Amaozn ECR](https://aws.amazon.com/ecr/) repository, and host our custom model behind a SageMaker endpoint to generate prediction.

SageMaker also provides several built in algorithms, for image classification, regression and clustering of structured data, timeseries processing and natural language processing. In the later part of this workshop we'll use [DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html), which is a supervised learning algorithm for forecasting one-dimensional time series using RNN.

## Disclaimer

This workshop is not an exercise in statistical methods, neither does it attempt to build a viable stock prediction model that you can use to make money. However it does showcase the techniques that you can use on AWS Machine Learning platform

## 1. Getting Started

Since you will execute most of the workshop steps on a Jupyter Notebook hosted on SageMaker, start by creating a notebook instance on SageMaker from AWS Console.

Refer to [AWS Region Table](https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/) to check the availability of SageMaker service, and choose to create the following infrastructure in any of the regions where it is available.
As of re:Invent-2018, SageMaker is available in the following regions:
- us-east-1 (Norther Virgina)
- us-east-2 (Ohio)
- us-west-2 (Oregon)
- eu-west-1 (Ireland)
- eu-central-1 (Frankfurt)
- ap-northeast-1 (Tokyo)
- ap-northeast-2 (Seoul)
- ap-southeast-2 (Sydney)


### 1.1. Lifecycle configuration
1.  Lifecycle configurations are small bootup scripts, that you can use to automate certain tasks when a Notebook instance in being created and/or being started. For this workshop, create a startup script to download pre-built notebooks from this Github repository onto your notebook instance. Configure this script to run on `Create notebook`.
    ```
    #!/bin/bash
    set -e
    git clone https://github.com/aws-samples/amazon-sagemaker-stock-prediction.git
    mkdir SageMaker/fsv309-workshop
    mv mazon-sagemaker-stock-prediction/container SageMaker/fsv309-workshop/container/
    mv mazon-sagemaker-stock-prediction/notebooks SageMaker/fsv309-workshop/notebooks/
    mv mazon-sagemaker-stock-prediction/pretrained-model SageMaker/fsv309-workshop/pretrained-model/
    rm -rf mazon-sagemaker-stock-prediction
    sudo chmod -R ugo+w SageMaker/fsv309-workshop/
    sudo yum install -y docker
    ```

1. Also create a  startup script as follows, and configure it to run on `Start Notebook`.

    ```
    #!/bin/bash
    set -e
    sudo service docker start
    ```
### 1.2. Notebook instance
1. Use the lifecycle configuration to create Notebook instance in a region of your choice.
1. Choose a moderatly sized memory optimized instance class, such as `ml.m4.xlarge`
1. If you do not have an IAM role created prior with all the necessary permissions needed for SageMaker to operate, create a new role on the fly.
1. Optionally you can choose to place your instance within a VPC and encrypt all data to be used within notebook to be encrypted. For the purpose fo the workshop you can proceed without these mechanisms to protect the notebook.

### 1.3. Athena Table
Athena allows you to query data directly from S3 buckets, using standard SQL compatible queries. Use the following DDL to create external table in athena, that will create schema and then allow queries to be rub directly on stock market data as stored in S3 buckets maintained by Deutsche Börse.

    ```
    CREATE EXTERNAL TABLE `xetra`(
    `isin` string COMMENT 'from deserializer', 
    `mnemonic` string COMMENT 'from deserializer', 
    `securitydesc` string COMMENT 'from deserializer', 
    `securitytype` string COMMENT 'from deserializer', 
    `currency` string COMMENT 'from deserializer', 
    `securityid` string COMMENT 'from deserializer', 
    `date` string COMMENT 'from deserializer', 
    `time` string COMMENT 'from deserializer', 
    `startprice` string COMMENT 'from deserializer', 
    `maxprice` string COMMENT 'from deserializer', 
    `minprice` string COMMENT 'from deserializer', 
    `endprice` string COMMENT 'from deserializer', 
    `tradedvolume` string COMMENT 'from deserializer', 
    `numberoftrades` string COMMENT 'from deserializer')
    ROW FORMAT SERDE 
    'org.apache.hadoop.hive.serde2.OpenCSVSerde' 
    WITH SERDEPROPERTIES ( 
    'quoteChar'='\"', 
    'separatorChar'=',', 
    'skip.header.line.count'='1') 
    STORED AS INPUTFORMAT 
    'org.apache.hadoop.mapred.TextInputFormat' 
    OUTPUTFORMAT 
    'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
    LOCATION
    's3://deutsche-boerse-xetra-pds/'
    TBLPROPERTIES (
    'classification'='csv', 
    'transient_lastDdlTime'='1540333010')
    ```

## 2. Data preparation

## 3. Data Visualization

## 4. Custom RNN

## 5. SageMaker DeepAR
