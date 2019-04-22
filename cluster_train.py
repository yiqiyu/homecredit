from mmlspark import LightGBMRegressor
from mmlspark import LightGBMRegressionModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

from feat_loading import *

app_train = load_dataframe("train_all2.csv")
app_test = load_dataframe("test_all2.csv")
app_train, app_test = load_extra_feats_post(app_train, app_test)
to_del = del_useless_cols(app_train)
app_test = app_test.drop(to_del, axis=1)


conf = SparkConf().setMaster("spark://master:7077").setAppName("MMLSPARK")
sc = SparkContext(conf = conf)

spark = SparkSession \
        .builder \
        .appName("Python Spark SQL Hive integration example") \
        .enableHiveSupport() \
        .getOrCreate()

app_train_spark = spark.createDataFrame(app_train)
app_test_spark = spark.createDataFrame(app_test)


model = LightGBMRegressor(boosting="goss", numIterations=10000, objective='binary',
        learningRate=0.005, seed=50,
        baggingFraction=0.87, lambdaL1=0.4, lambdaL2=0.4, minDataInLeaf=30,
        maxDepth=9, featureFraction=0.66, numLeaves=47)


model.saveNativeModel("mymodel")
model = LightGBMRegressionModel.loadNativeModelFromFile("mymodel")