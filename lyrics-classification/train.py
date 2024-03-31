import shutil
import os
import warnings
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    StopWordsRemover,
    Word2Vec,
    StringIndexer,
    RegexTokenizer
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as fun

warnings.filterwarnings("ignore", category=UserWarning)
# Create a SparkSession
spark = SparkSession.builder.appName("LyricGenreClassification").getOrCreate()
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

# Load the CSV data
data = spark.read.format("csv") \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("maxRowsInMemory", 1000000) \
    .load("../dataset/Mendeley dataset.csv") \
    .select("lyrics", "genre") \
    .withColumn("lyrics", fun.lower(fun.col("lyrics")))  # Convert to lowercase for case-insensitivity

# Filter for desired genres
data = data.filter(fun.col("genre").isin(["pop", "country", "blues", "jazz", "reggae", "rock", "hip hop"]))

# Encode labels
indexer = StringIndexer(inputCol="genre", outputCol="label")
data = indexer.fit(data).transform(data)

# Select only necessary columns
data = data.select("label", "lyrics")


# Create preprocessing stages
regexTokenizer = RegexTokenizer(inputCol="lyrics", outputCol="words", pattern=r'\s+|,')
remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
word2Vec = Word2Vec(inputCol="filteredWords", outputCol="features")

# Create the model (LogisticRegression)
lr = LogisticRegression(maxIter=5, regParam=0.01, labelCol="label", featuresCol="features")

# Create the pipeline
pipeline = Pipeline(stages=[regexTokenizer, remover, word2Vec, lr])

# Define parameter grid for cross-validation
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [5]) \
    .addGrid(lr.regParam, [0.01]) \
    .build()

# Define evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

# Create cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Fit the model with cross-validation
history = cv.fit(data)

model = history.bestModel

output_dir = "../model/trained_model"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

model.save(output_dir)

# Evaluate best model
predictions = model.transform(data)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Stop the SparkSession
spark.stop()