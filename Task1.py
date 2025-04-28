from pyspark.sql import SparkSession

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("FakeNews_Load_Exploration") \
    .getOrCreate()

# Step 2: Load CSV and infer schema
file_path = "fake_news_sample.csv"  # Adjusted for your upload location
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Step 3: Create Temporary View
df.createOrReplaceTempView("news_data")

# Step 4: Basic Queries

# Show first 5 rows
df.show(5, truncate=False)

# Count total number of articles
total_articles = spark.sql("SELECT COUNT(*) AS total_articles FROM news_data")
total_articles.show()

# Retrieve distinct labels (FAKE or REAL)
distinct_labels = spark.sql("SELECT DISTINCT label FROM news_data")
distinct_labels.show()

# Step 5: Save output to CSV
# Saving the first 5 rows to task1_output.csv
df.limit(5).coalesce(1).write.csv("Outputs/Task1_Output", header=True, mode='overwrite')
