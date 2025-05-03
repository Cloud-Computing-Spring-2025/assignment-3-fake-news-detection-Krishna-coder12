# Assignment-5-FakeNews-Detection

## **Overview**
In this assignment, you are going to build a simple machine learning pipeline using Spark MLlib to classify news articles as FAKE or REAL based on their content. You will clean and process the text data, extract features, train a basic model, and evaluate its accuracy.

---
## Dataset File
- **File Name**: `fake_news_sample.csv`
- **Total Records**: 500
- **Total Columns**: 4

## Column Descriptions

| Column Name | Description |
|-------------|-------------|
| `id`        | Unique identifier for each article. |
| `title`     | Title of the news article. |
| `text`      | Full text content of the article. |
| `label`     | Ground truth label: `REAL` or `FAKE`. |

---
## **Prerequisites**

Before starting the assignment, ensure you have the following software installed and properly configured on your machine:

1. **Python 3.x**:
   - [Download and Install Python](https://www.python.org/downloads/)
   - Verify installation:
     ```bash
     python3 --version
     ```

2. **PySpark**:
   - Install using `pip`:
     ```bash
     pip install pyspark
     ```

3. **Apache Spark**:
   - Ensure Spark is installed. You can download it from the [Apache Spark Downloads](https://spark.apache.org/downloads.html) page.
   - Verify installation by running:
     ```bash
     spark-submit --version
     ```

4. **Faker**:
   - Install using `pip`:
     ```bash
     pip install faker
     ```
---
## 🔹 Task 1: Load & Basic Exploration
**Objective:**  
Load the dataset, handle missing values, and prepare the data for model training.

**Steps:**
- Load the dataset using `pandas`.
- Remove unnecessary columns (e.g., 'id').
- Fill missing values in the 'text' column with empty strings.
- Concatenate 'title', 'author', and 'text' columns to create a unified `content` feature.
- Create a binary label from the 'label' column (1 = fake, 0 = real).

---

## 🔹 Task 2: Text Preprocessing
**Objective:**  
Convert the unified text column into numerical vectors using TF-IDF.

**Steps:**
- Use `TfidfVectorizer` from `sklearn.feature_extraction.text`.
- Limit the max number of features (e.g., 5000).
- Fit-transform the combined `content` column to get feature vectors.
- Store the resulting TF-IDF matrix as X and the labels as y.

---

## 🔹 Task 3: Feature Extraction
**Objective:**  
Train a Logistic Regression model and evaluate its performance.

**Steps:**
- Split the TF-IDF vectorized data into training and test sets (80/20).
- Train a `LogisticRegression` classifier.
- Evaluate the model using accuracy score, confusion matrix, and classification report.
- Print and interpret the evaluation results.

---

## 🔹 Task 4: Model Training
**Objective:**  
Train and compare multiple classifiers to evaluate performance beyond Logistic Regression.

**Steps:**
- Train the following models:
  - LogisticRegression
  - RandomForestClassifier
  - PassiveAggressiveClassifier
- Evaluate each model on accuracy.
- Print a table showing model names and their respective accuracies.
- Identify the best performing model based on test set performance.

---

## 🔹 Task 5: Evaluate the Model
**Objective:**  
Deploy the fake news detection model via a simple web interface using Streamlit.

**Steps:**
- Load the pre-trained TF-IDF vectorizer and classifier.
- Design a Streamlit web UI with a text input box.
- When a user enters news text, convert it using TF-IDF and predict using the model.
- Display whether the news is Real or Fake in the UI.
---

## **Setup Instructions**

### **1. Running the Analysis Tasks**

You can run the analysis tasks either locally or using Docker.

1. **Generate the python file**:
   ```bash
   python data_generator.py
   ```

2. **Run Your PySpark Scripts Using `spark-submit`**:
   - Once you perform all the three tasks, open a new terminal and run the following:
   ```bash
   spark-submit Task1.py
   spark-submit Task2.py
   spark-submit Task3.py
   spark-submit Task4.py
   spark-submit Task5.py
   ```

3. **Verify the Outputs**:
   On your host machine, check the `output/` directory for the resulting files.
---

