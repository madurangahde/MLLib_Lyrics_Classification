from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


def predict_genre(spark, model, lyric):
    """
    Predict the genre for a single lyric using the saved model and convert numerical predictions to strings.

    Args:
        model_path: Path to the saved PipelineModel
        lyric: Single lyric string

    Returns:
        The predicted genre as a string
    """

    # Create a DataFrame with the single lyric (genre is a placeholder, not used for prediction)
    data = [(lyric, "unknown")]
    df = spark.createDataFrame(data, ["lyrics", "genre"])

    # Make predictions
    predictions = model.transform(df)

    # Extract prediction and probabilities (adjust based on your model's output)
    result = predictions.select("prediction", "probability").collect()[0]
    predicted_prob = max(result["probability"])  # Assuming probability is a vector
    genre_prob_dict = dict(enumerate(result["probability"]))  # Mapping to genre indices

    # Extract the StringIndexerModel from the pipeline (5th stage, index 4)
    indexer_model = model.stages[4]
    genre_labels = indexer_model.labels

    # Define a UDF to map numerical predictions to genre strings
    def index_to_genre(index):
        return genre_labels[int(index)]

    index_to_genre_udf = udf(index_to_genre, StringType())

    # Add a new column with the predicted genre as a string
    predictions = predictions.withColumn(
        "predicted_genre", index_to_genre_udf(col("prediction"))
    )

    # Collect the result
    result = predictions.select("predicted_genre").first()
    predicted_genre = result["predicted_genre"]

    # Stop Spark session
    spark.stop()

    return predicted_genre, predicted_prob, genre_prob_dict


# Example usage
# if __name__ == "__main__":
#     model_path = "/home/joel-sathiyendra/University/BigDataAnalytics/labs/Lyric-Classifier/front-end/genre_prediction_model_2"
#     lyric = "I got a feeling that tonight's gonna be a good night"
#     predicted_genre, predicted_prob, genre_prob_dict = predict_genre(model_path, lyric)
#     print(f"Predicted Genre: {predicted_genre}")
#     print(f"Confidence: {predicted_prob:.4f}")
#     print("Genre Probabilities:", genre_prob_dict)
