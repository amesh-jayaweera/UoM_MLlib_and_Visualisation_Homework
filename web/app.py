# import os
from pyspark.sql import SparkSession
from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import Row
# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

app = Flask(__name__)

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LyricGenreClassification") \
    .getOrCreate()

# Load the trained model
model = PipelineModel.load("model/trained_model")


def predict_genre(lyrics):
    # Create a Spark DataFrame directly from the provided lyrics
    spark_df = spark.createDataFrame([Row(lyrics=lyrics)])
    # Make prediction
    prediction = model.transform(spark_df)
    return prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    lyrics = request.form['lyrics']
    prediction = predict_genre(lyrics)

    # Retrieve genre names from the model metadata
    metadata = model.stages[-1].metadata
    genres = metadata["label"].metadata["ml_attr"]["vals"]

    # Extract predicted probabilities
    probs = prediction.select("probability").collect()[0][0].toArray()

    # Pass genre names and probabilities to the HTML template
    return render_template('result.html', genres=genres, probs=probs)


if __name__ == '__main__':
    app.run(debug=True)
