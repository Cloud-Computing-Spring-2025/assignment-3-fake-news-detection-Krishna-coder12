from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Step 0: Create Spark Session (if not already created)
spark = SparkSession.builder \
    .appName("FakeNews_ModelEvaluation") \
    .getOrCreate()

# Step 1: Load predictions from Task 4 (if not already available)
predictions = spark.read.csv("Outputs/Task4_Output", header=True, inferSchema=True)

# Step 2: Evaluate Metrics

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator_acc.evaluate(predictions)

# F1 Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)
f1_score = evaluator_f1.evaluate(predictions)

# Step 3: Create a DataFrame for the metrics
metrics_data = [
    ("Accuracy", round(accuracy, 2)),
    ("F1 Score", round(f1_score, 2))
]

metrics_df = spark.createDataFrame(metrics_data, ["Metric", "Value"])

# Step 4: Save to CSV
metrics_df.coalesce(1).write.csv("Outputs/Task5_Output", header=True, mode='overwrite')
