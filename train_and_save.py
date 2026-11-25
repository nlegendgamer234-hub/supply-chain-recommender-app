import os, re, sys, time

# Windows fix + use this venv's Python
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

BEST_DEPTH = 3   # chosen on the validation set in the earlier run

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

spark = (SparkSession.builder.appName("SupplyChainTrainSave").master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.default.parallelism", "8")
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = (spark.read.option("header", True).option("inferSchema", True)
      .option("encoding", "ISO-8859-1").csv("data/DataCoSupplyChainDataset.csv"))
df = df.toDF(*[re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns])

ts = F.to_timestamp("order_date_dateorders", "M/d/yyyy H:mm")
df = (df.withColumn("order_month", F.month(ts))
        .withColumn("order_weekday", F.dayofweek(ts))
        .withColumn("order_hour", F.hour(ts)))

label = "late_delivery_risk"
cat_cols = ["shipping_mode", "order_region", "customer_segment", "market",
            "type", "department_name", "category_name"]
num_cols = ["days_for_shipment_scheduled", "order_item_quantity", "order_item_discount_rate",
            "order_item_product_price", "order_month", "order_weekday", "order_hour"]

data = (df.select(cat_cols + num_cols + [label]).na.drop()
          .withColumn(label, F.col(label).cast("double"))).cache()
log(f"Rows: {data.count()}")

train, test = data.randomSplit([0.8, 0.2], seed=42)

indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
features = [c + "_idx" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
dt = DecisionTreeClassifier(labelCol=label, featuresCol="features",
                            maxDepth=BEST_DEPTH, maxBins=64, seed=42)

model = Pipeline(stages=indexers + [assembler, dt]).fit(train)
log("Model trained")

pred = model.transform(test)
acc = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy").evaluate(pred)
f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1").evaluate(pred)
auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC").evaluate(pred)
print(f"\nTEST  accuracy={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
pred.groupBy(label, "prediction").count().orderBy(label, "prediction").show()

os.makedirs("models", exist_ok=True)
with open("models/tree_rules.txt", "w", encoding="utf-8") as f:
    f.write(model.stages[-1].toDebugString)

model.write().overwrite().save("models/dt_pipeline")
log("MODEL SAVED to models/dt_pipeline")

spark.stop()