import json
import recommend as rec

# The tree only uses shipping mode and order hour, so we ask the Spark model
# for its late-delivery probability for every mode at every hour of the day.
base = {
    "order_region": "Western Europe", "customer_segment": "Consumer", "market": "Africa",
    "type": "CASH", "department_name": "Golf", "category_name": "Accessories",
    "order_item_quantity": 1.0, "order_item_discount_rate": 0.05,
    "order_item_product_price": 100.0, "order_month": 6.0, "order_weekday": 4.0,
}

spark = rec.get_spark()
spark.sparkContext.setLogLevel("ERROR")
model = rec.load_model()

table = {}
for hour in range(24):
    order = dict(base)
    order["order_hour"] = float(hour)
    table[str(hour)] = rec.score_modes(spark, model, order)
    print("hour", hour, "done", flush=True)

with open("risk_table.json", "w", encoding="utf-8") as f:
    json.dump(table, f, indent=1)
print("risk_table.json created")
spark.stop()