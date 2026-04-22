from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    StringIndexer,
    CountVectorizer,
)
from pyspark.ml.classification import NaiveBayes
from nltk.stem import SnowballStemmer


class StemmerTransformer(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Custom transformer to apply stemming to an array of words.
    """

    def __init__(self, inputCol=None, outputCol=None):
        super(StemmerTransformer, self).__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)
        self.stemmer = SnowballStemmer("english")

    def _transform(self, dataset):
        def stem_words(words):
            if words is None:
                return None
            return [self.stemmer.stem(word) for word in words]

        stem_udf = udf(stem_words, ArrayType(StringType()))
        return dataset.withColumn(
            self.getOutputCol(), stem_udf(col(self.getInputCol()))
        )


class WordArrayToStringTransformer(
    Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
):
    """
    Custom transformer to convert an array of words to a single string.
    """

    def __init__(self, inputCol=None, outputCol=None):
        super(WordArrayToStringTransformer, self).__init__()
        self._set(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, dataset):
        def unite_words(words):
            if words is None:
                return None
            return " ".join(words)

        unite_udf = udf(unite_words, StringType())
        return dataset.withColumn(
            self.getOutputCol(), unite_udf(col(self.getInputCol()))
        )


def build_pipeline():
    """
    Build the Spark ML pipeline for music genre prediction.

    Returns:
        Pipeline object with all stages configured
    """
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    stemmer = StemmerTransformer(inputCol="filtered_words", outputCol="stemmed_words")
    uniter = WordArrayToStringTransformer(
        inputCol="stemmed_words", outputCol="processed_lyrics"
    )
    label_indexer = StringIndexer(inputCol="genre", outputCol="label")
    countVectors = CountVectorizer(
        inputCol="stemmed_words", outputCol="features", vocabSize=10000, minDF=5
    )
    nb = NaiveBayes(smoothing=1)
    pipeline = Pipeline(
        stages=[tokenizer, remover, stemmer, uniter, label_indexer, countVectors, nb]
    )
    return pipeline
