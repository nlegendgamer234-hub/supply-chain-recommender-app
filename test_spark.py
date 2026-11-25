from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").master("local[*]").getOrCreate()
df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "name"])
df.show()
spark.stop()