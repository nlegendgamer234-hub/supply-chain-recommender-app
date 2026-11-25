import re, os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

spark = SparkSession.builder.appName("SupplyChainTrain").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = (spark.read.option("header", True).option("inferSchema", True)
      .option("encoding", "ISO-8859-1").csv("data/DataCoSupplyChainDataset.csv"))
df = df.toDF(*[re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns])

# Time features from the order date
ts = F.to_timestamp("order_date_dateorders", "M/d/yyyy H:mm")
df = (df.withColumn("order_month", F.month(ts))
        .withColumn("order_weekday", F.dayofweek(ts))
        .withColumn("order_hour", F.hour(ts)))

label = "late_delivery_risk"
# Only information available when the order is placed (no leakage)
cat_cols = ["shipping_mode", "order_region", "customer_segment", "market",
            "type", "department_name", "category_name"]
num_cols = ["days_for_shipment_scheduled", "order_item_quantity", "order_item_discount_rate",
            "order_item_product_price", "order_month", "order_weekday", "order_hour"]

data = (df.select(cat_cols + num_cols + [label]).na.drop()
          .withColumn(label, F.col(label).cast("double")))
print("Rows used for training:", data.count())

train, val, test = data.randomSplit([0.7, 0.15, 0.15], seed=42)

# Feature pipeline: text columns -> numbers, then combine into one vector
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
features = [c + "_idx" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
prep = Pipeline(stages=indexers + [assembler]).fit(train)

train_p, val_p, test_p = [prep.transform(x).cache() for x in (train, val, test)]

acc = MulticlassClassificationEvaluator(labelCol=label, metricName="accuracy")
f1 = MulticlassClassificationEvaluator(labelCol=label, metricName="f1")
auc = BinaryClassificationEvaluator(labelCol=label, metricName="areaUnderROC")

# Tune tree depth on the validation set
best, best_acc, best_depth = None, 0, None
for d in [3, 5, 7, 10, 12]:
    m = DecisionTreeClassifier(labelCol=label, featuresCol="features",
                               maxDepth=d, maxBins=64, seed=42).fit(train_p)
    a = acc.evaluate(m.transform(val_p))
    print(f"maxDepth={d}  validation accuracy={a:.4f}")
    if a > best_acc:
        best, best_acc, best_depth = m, a, d
print("Best maxDepth:", best_depth)

# Final evaluation on the untouched test set
pred = best.transform(test_p)
n1 = test.filter(F.col(label) == 1).count()
n0 = test.filter(F.col(label) == 0).count()
print("\n=== TEST RESULTS ===")
print("Majority-class baseline:", round(max(n1, n0) / (n1 + n0), 4))
print("Accuracy:", round(acc.evaluate(pred), 4))
print("F1 score:", round(f1.evaluate(pred), 4))
print("AUC:", round(auc.evaluate(pred), 4))
pred.groupBy(label, "prediction").count().orderBy(label, "prediction").show()

print("Top feature importances:")
imps = sorted(zip(features, best.featureImportances.toArray()), key=lambda x: -x[1])[:10]
for name, imp in imps:
    print(f"  {name:32s} {imp:.4f}")

os.makedirs("models", exist_ok=True)
with open("models/tree_rules.txt", "w") as f:
    f.write("Features in order: " + str(features) + "\n\n" + best.toDebugString)

# Save the full pipeline (feature steps + tree)
final = PipelineModel(prep.stages + [best])
try:
    final.write().overwrite().save("models/dt_pipeline")
    print("\nMODEL SAVED to models/dt_pipeline")
except Exception as e:
    print("\nMODEL SAVE FAILED:", str(e)[:300])

spark.stop()