from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import lower, col
from pyspark.sql.functions import concat_ws, concat, lit, udf
from pyspark.sql.types import StringType

# Step 0: Create Spark Session
spark = SparkSession.builder \
    .appName("FakeNews_ModelTraining") \
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

# Step 6: Label Indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
df_indexed = indexer.fit(df_tfidf).transform(df_tfidf)

# Step 7: Split into Training and Test sets (80/20 split)
train_data, test_data = df_indexed.randomSplit([0.8, 0.2], seed=42)

# Step 8: Train Logistic Regression Model
lr = LogisticRegression(featuresCol='features', labelCol='label_index')
lr_model = lr.fit(train_data)

# Step 9: Predict on Test Data
predictions = lr_model.transform(test_data)

# Step 10: Select required columns for output
predictions_output = predictions.select("id", "title", "label_index", "prediction")

# Step 11: Save Predictions to CSV
predictions_output.coalesce(1).write.csv("Outputs/Task4_Output", header=True, mode='overwrite')
