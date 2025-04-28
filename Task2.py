from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import lower, col, concat_ws, concat, lit

# Step 0: Create Spark Session (if not already created)
spark = SparkSession.builder \
    .appName("FakeNews_TextPreprocessing") \
    .getOrCreate()

# Step 1: Load CSV and infer schema
file_path = "fake_news_sample.csv"  # adjust your path if needed
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 2: Lowercase the 'text' column
df_lower = df.withColumn("text", lower(col("text")))

# Step 3: Tokenize the text column into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_tokenized = tokenizer.transform(df_lower)

# Step 4: Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokenized)

# Step 5: Convert filtered_words array to string formatted like [word1, word2, ...]
df_final = df_filtered.withColumn(
    "filtered_words",
    concat(
        lit("["), 
        concat_ws(", ", col("filtered_words")),
        lit("]")
    )
)

# Step 6: Select necessary columns
df_task2 = df_final.select("id", "title", "filtered_words", "label")

# (Optional) Step 7: Create Temporary View
df_task2.createOrReplaceTempView("cleaned_news")

# Step 8: Save output to CSV
df_task2.coalesce(1).write.csv("Outputs/Task2_Output", header=True, mode='overwrite')
