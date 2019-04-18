from mmlspark import LightGBMRegressor
from mmlspark import LightGBMRegressionModel
from pyspark import SparkConf, SparkContext


conf = SparkConf().setMaster("spark://master:7077").setAppName("MMLSPARK")
sc = SparkContext(conf = conf)

model = LightGBMRegressor(boosting="goss", numIterations=10000, objective='binary',
        learningRate=0.005, seed=50,
        baggingFraction=0.87, lambdaL1=0.4, lambdaL2=0.4, minDataInLeaf=30,
        maxDepth=9, featureFraction=0.66, numLeaves=47)


model.saveNativeModel("mymodel")
model = LightGBMRegressionModel.loadNativeModelFromFile("mymodel")