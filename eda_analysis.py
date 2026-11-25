import re, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SupplyChainEDA").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = (spark.read.option("header", True).option("inferSchema", True)
      .option("encoding", "ISO-8859-1").csv("data/DataCoSupplyChainDataset.csv"))
df = df.toDF(*[re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower() for c in df.columns])
df.createOrReplaceTempView("orders")

os.makedirs("charts", exist_ok=True)

def run(title, sql):
    print("\n=== " + title + " ===")
    res = spark.sql(sql)
    res.show(15, truncate=False)
    return res.collect()

def bar(rows, xcol, ycol, title, fname):
    plt.figure(figsize=(8, 4.5))
    plt.bar([str(r[xcol]) for r in rows], [r[ycol] for r in rows], color="#3b6fb6")
    plt.title(title); plt.ylabel(ycol); plt.xticks(rotation=25, ha="right")
    plt.tight_layout(); plt.savefig("charts/" + fname, dpi=150); plt.close()

# Q1 Late delivery rate by shipping mode
r1 = run("Late delivery % by shipping mode", """
    SELECT shipping_mode, COUNT(*) AS orders,
           ROUND(AVG(late_delivery_risk)*100,1) AS late_pct
    FROM orders GROUP BY shipping_mode ORDER BY late_pct DESC""")
bar(r1, "shipping_mode", "late_pct", "Late delivery % by shipping mode", "late_by_shipping_mode.png")

# Q2 Late delivery rate by order region
r2 = run("Late delivery % by order region", """
    SELECT order_region, COUNT(*) AS orders,
           ROUND(AVG(late_delivery_risk)*100,1) AS late_pct
    FROM orders GROUP BY order_region ORDER BY late_pct DESC""")
bar(r2, "order_region", "late_pct", "Late delivery % by region", "late_by_region.png")

# Q3 Late delivery rate by customer segment
r3 = run("Late delivery % by customer segment", """
    SELECT customer_segment, COUNT(*) AS orders,
           ROUND(AVG(late_delivery_risk)*100,1) AS late_pct
    FROM orders GROUP BY customer_segment ORDER BY late_pct DESC""")

# Q4 Scheduled vs real shipping days by mode
r4 = run("Average scheduled vs real shipping days", """
    SELECT shipping_mode,
           ROUND(AVG(days_for_shipment_scheduled),2) AS scheduled_days,
           ROUND(AVG(days_for_shipping_real),2) AS real_days
    FROM orders GROUP BY shipping_mode ORDER BY real_days DESC""")
x = range(len(r4)); w = 0.38
plt.figure(figsize=(8, 4.5))
plt.bar([i - w/2 for i in x], [r["scheduled_days"] for r in r4], w, label="Scheduled")
plt.bar([i + w/2 for i in x], [r["real_days"] for r in r4], w, label="Real")
plt.xticks(list(x), [r["shipping_mode"] for r in r4], rotation=25, ha="right")
plt.title("Scheduled vs real shipping days"); plt.legend(); plt.tight_layout()
plt.savefig("charts/scheduled_vs_real_days.png", dpi=150); plt.close()

# Q5 Delivery status vs late risk (shows target leakage)
run("Delivery status vs late_delivery_risk", """
    SELECT delivery_status, late_delivery_risk, COUNT(*) AS orders
    FROM orders GROUP BY delivery_status, late_delivery_risk
    ORDER BY delivery_status, late_delivery_risk""")

# Q6 Top 10 product categories by sales
run("Top 10 categories by sales", """
    SELECT category_name, ROUND(SUM(sales),0) AS total_sales
    FROM orders GROUP BY category_name ORDER BY total_sales DESC LIMIT 10""")

spark.stop()
print("\nCharts saved in the charts/ folder")