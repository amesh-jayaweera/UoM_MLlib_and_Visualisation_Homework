from nltk.stem import PorterStemmer
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as fun


class Stemmer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, outputCol=None):
        super(Stemmer, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)
        self.stemmer = PorterStemmer()

    def _transform(self, df):
        stemmer_udf = fun.udf(lambda tokens: [self.stemmer.stem(token) for token in tokens], ArrayType(StringType()))
        return df.withColumn(self.getOutputCol(), stemmer_udf(df[self.getInputCol()]))
