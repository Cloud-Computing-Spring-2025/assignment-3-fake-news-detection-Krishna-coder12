from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.sql.functions import lower, col, concat_ws, concat, lit, udf
from pyspark.sql.types import StringType

# Step 0: Create Spark Session
spark = SparkSession.builder \
    .appName("FakeNews_FeatureExtraction") \
    .getOrCreate()

# Step 1: Load CSV
file_path = "fake_news_sample.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 2: Lowercase text
df_lower = df.withColumn("text", lower(col("text")))

# Step 3: Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_tokenized = tokenizer.transform(df_lower)

# Step 4: Remove Stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Step 5: TF-IDF Feature Extraction
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
df_hashed = hashingTF.transform(df_filtered)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(df_hashed)
df_tfidf = idf_model.transform(df_hashed)

# Step 6: Label Indexing (FAKE → 0, REAL → 1)
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_indexed = indexer.fit(df_tfidf).transform(df_tfidf)

# Step 7: Convert 'filtered_words' array into string with [word1, word2, ...]
df_with_filtered_words_str = df_indexed.withColumn(
    "filtered_words_str",
    concat(lit("["), concat_ws(", ", col("filtered_words")), lit("]"))
)

# Step 8: Convert 'features' vector into string
def vector_to_string(v):
    return str(v)

vector_to_string_udf = udf(vector_to_string, StringType())

df_final = df_with_filtered_words_str.withColumn(
    "features_str",
    vector_to_string_udf(col("features"))
)

# Step 9: Select only required columns
df_task3 = df_final.select("id", "filtered_words_str", "features_str", "label_index")

# Step 10: Rename columns to match output
df_task3 = df_task3.withColumnRenamed("filtered_words_str", "filtered_words") \
                   .withColumnRenamed("features_str", "features")

# Step 11: Save output to CSV
df_task3.coalesce(1).write.csv("Outputs/Task3_Output", header=True, mode='overwrite')
