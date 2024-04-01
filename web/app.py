import nltk
from pyspark.sql import SparkSession
from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import Row


app = Flask(__name__)

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LyricGenreClassification") \
    .getOrCreate()

nltk.download('punkt')

# Load the trained model
model = PipelineModel.load("./model/trained_model")


def predict_genre(lyrics):
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

    # Extract probabilities
    probabilities = prediction.select("probability").collect()[0][0]

    # Map probabilities to genre labels
    genre_labels = model.stages[-3].labels
    genre_probabilities = {genre_labels[i]: float(probabilities[i]) for i in range(len(genre_labels))}

    print(genre_probabilities)

    # Pass genre names and probabilities to the HTML template
    return render_template('result.html', predictions=genre_probabilities)


if __name__ == '__main__':
    app.run(debug=True)
