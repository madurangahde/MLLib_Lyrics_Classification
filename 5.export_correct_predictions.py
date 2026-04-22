import argparse
import os
import shutil
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def normalize_genre(col_name: str):
    """Normalize genre text so filtering/comparison is consistent."""
    cleaned = F.lower(F.trim(F.col(col_name)))
    cleaned = F.regexp_replace(cleaned, r"[-_]+", " ")
    cleaned = F.regexp_replace(cleaned, r"\s+", " ")
    return cleaned


def ensure_parent_dir(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)


def write_single_csv(df, output_file: Path):
    """Write DataFrame as one CSV file (not a folder with part files)."""
    temp_dir = output_file.parent / f".__tmp_{output_file.stem}"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(temp_dir))

    part_files = list(temp_dir.glob("part-*.csv"))
    if not part_files:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError("No CSV part file produced in temporary directory.")

    if output_file.exists():
        output_file.unlink()

    shutil.move(str(part_files[0]), str(output_file))
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run genre predictions with a saved Spark pipeline model and export only "
            "correct predictions for selected genres to a CSV file."
        )
    )
    parser.add_argument(
        "--dataset",
        default="data/Merged_dataset.csv",
        help="Path to input dataset CSV (default: data/Merged_dataset.csv)",
    )
    parser.add_argument(
        "--model",
        default="frontend/model_stage4_merged",
        help="Path to saved Spark PipelineModel (default: frontend/model_stage4_merged)",
    )
    parser.add_argument(
        "--output",
        default="data/correct_predictions_selected_genres.csv",
        help="Output CSV file path (default: data/correct_predictions_selected_genres.csv)",
    )
    args = parser.parse_args()

    target_genres = ["country", "blues", "jazz", "reggae", "hip hop"]

    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    ensure_parent_dir(output_path)

    # Java 21+ can throw UnsupportedOperationException for Subject.getSubject().
    # This JVM flag keeps behavior compatible with Spark/Hadoop in local runs.
    os.environ.setdefault("SPARK_SUBMIT_OPTS", "-Djava.security.manager=allow")

    spark = (
        SparkSession.builder.appName("Export Correct Genre Predictions")
        .master("local[*]")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
        .getOrCreate()
    )

    try:
        model = PipelineModel.load(str(model_path))

        df = (
            spark.read.option("header", True)
            .option("multiLine", True)
            .option("escape", '"')
            .option("quote", '"')
            .csv(str(dataset_path))
        )

        required_cols = {"genre", "lyrics"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        df = df.withColumn("genre_norm", normalize_genre("genre"))

        # Restrict evaluation to requested genres only.
        filtered = df.filter(F.col("genre_norm").isin(target_genres))

        predictions = model.transform(filtered)

        indexer_model = model.stages[4]
        labels = indexer_model.labels

        def index_to_genre(index):
            if index is None:
                return None
            idx = int(index)
            return labels[idx] if 0 <= idx < len(labels) else None

        index_to_genre_udf = F.udf(index_to_genre, StringType())

        predictions = (
            predictions.withColumn("predicted_genre", index_to_genre_udf(F.col("prediction")))
            .withColumn("predicted_genre_norm", normalize_genre("predicted_genre"))
            .withColumn("prob_array", vector_to_array(F.col("probability")))
            .withColumn("predicted_confidence", F.array_max(F.col("prob_array")))
        )

        correct = predictions.filter(F.col("genre_norm") == F.col("predicted_genre_norm"))

        output_df = correct.select(
            *[c for c in ["artist_name", "track_name", "release_date"] if c in correct.columns],
            "genre",
            "predicted_genre",
            "predicted_confidence",
            "lyrics",
        )

        write_single_csv(output_df, output_path)

        total_filtered = filtered.count()
        total_correct = correct.count()
        accuracy = (total_correct / total_filtered) if total_filtered else 0.0

        print(f"Input dataset      : {dataset_path}")
        print(f"Model path         : {model_path}")
        print(f"Filtered genres    : {', '.join(target_genres)}")
        print(f"Rows in scope      : {total_filtered}")
        print(f"Correct predictions: {total_correct}")
        print(f"Accuracy in scope  : {accuracy:.2%}")
        print(f"Output CSV         : {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
