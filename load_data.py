import re
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("SupplyChainLoad").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# This dataset is not UTF-8, so the encoding option is needed
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .option("encoding", "ISO-8859-1")
      .csv("data/DataCoSupplyChainDataset.csv"))

# Clean column names: "Days for shipping (real)" -> "days_for_shipping_real"
clean = [re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns]
df = df.toDF(*clean)

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.show(5, truncate=25)

nulls = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first().asDict()
print("Columns with nulls:", {k: v for k, v in nulls.items() if v > 0})

df.groupBy("late_delivery_risk").count().show()
df.groupBy("shipping_mode").count().show()

spark.stop()