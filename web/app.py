from pyspark.sql import SparkSession
from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import Row
import pandas as pd

app = Flask(__name__)

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LyricGenreClassification") \
    .getOrCreate()

# Load the trained model
model = PipelineModel.load("./model/trained_model")  # Update the path here

data = pd.read_csv("/Users/ameshmjayaweera/Documents/UoM_MLlib_and_Visualisation_Homework/dataset/Mendeley dataset.csv")


def get_key_by_max_value(d):
    max_val = max(d.values())
    print(max_val)
    if max_val >= 0.5:
        return max(d, key=d.get)
    else:
        return None


def test(lyrics):
    prediction = predict_genre(lyrics)

    # Extract probabilities
    probabilities = prediction.select("probability").collect()[0][0]

    # Map probabilities to genre labels
    genre_labels = model.stages[-3].labels
    genre_probabilities = {genre_labels[i]: float(probabilities[i]) for i in range(len(genre_labels))}

    predicted_genre = get_key_by_max_value(genre_probabilities)

    return predicted_genre

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

    # Extract probabilities
    probabilities = prediction.select("probability").collect()[0][0]

    # Map probabilities to genre labels
    genre_labels = model.stages[-3].labels
    genre_probabilities = {genre_labels[i]: float(probabilities[i]) for i in range(len(genre_labels))}

    print(genre_probabilities)

    predicted_genre = get_key_by_max_value(genre_probabilities)

    # Pass genre names and probabilities to the HTML template
    return render_template('result.html', predictions=genre_probabilities, predicted_genre=predicted_genre)


for index, row in data.iterrows():
    lyrics = row['lyrics']
    actual_genre = row['genre']

    # Prediction
    prediction = test(lyrics)
    print(prediction)

    # Check if prediction is not None and equal to actual genre
    if prediction is not None and prediction == actual_genre:
        print(actual_genre, lyrics)


if __name__ == '__main__':
    app.run(debug=True)
