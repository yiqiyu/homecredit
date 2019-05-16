from mmlspark import LightGBMRegressor
from mmlspark import LightGBMRegressionModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import os


conf = SparkConf().setMaster("spark://master:7077").setAppName("MMLSPARK")
sc = SparkContext(conf = conf)

# spark-submit --py-files /mnt/d/data_analysis/homecredit2/feat_loading.py
from feat_loading import *

# app_train = load_dataframe("/mnt/d/data_analysis/homecredit2/train_all2.csv")
# app_test = load_dataframe("/mnt/d/data_analysis/homecredit2/test_all2.csv")
# app_train, app_test = load_extra_feats_post(app_train, app_test)
# to_del = del_useless_cols(app_train)
# app_test = app_test.drop(to_del, axis=1)


spark = SparkSession \
        .builder \
        .appName("MMLSPARK") \
        .enableHiveSupport() \
        .getOrCreate()

# app_train_spark = spark.createDataFrame(app_train).collect()
# app_test_spark = spark.createDataFrame(app_test).collect()


app_train = spark.read.csv("/homeredit/train_all2.csv", header='true', inferSchema='true')
app_test = spark.read.csv("/homeredit/test_all2.csv", header='true', inferSchema='true')


model = LightGBMRegressor(boostingType="goss", numIterations=10000, objective='binary',
        learningRate=0.005, baggingSeed=50,
        baggingFraction=0.87, lambdaL1=0.4, lambdaL2=0.4, minSumHessianInLeaf=0.003,
        maxDepth=9, featureFraction=0.66, numLeaves=47)

model.fit(app_train)

model.saveNativeModel("mymodel")
model = LightGBMRegressionModel.loadNativeModelFromFile("mymodel")
