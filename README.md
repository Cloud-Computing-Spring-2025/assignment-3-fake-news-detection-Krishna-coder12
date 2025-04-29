# Assignment-5-FakeNews-Detection

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
   spark-submit task1.py
   spark-submit task2.py
   spark-submit task3.py
   ```

3. **Verify the Outputs**:
   On your host machine, check the `output/` directory for the resulting files.
---
