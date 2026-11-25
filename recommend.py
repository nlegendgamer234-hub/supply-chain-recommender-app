import os, sys

# Windows-only setup (on the website this part is skipped)
if os.name == "nt":
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]
    _jdk = r"C:\Program Files\Java\jdk1.8.0_202"
    if "JAVA_HOME" not in os.environ and os.path.isdir(_jdk):
        os.environ["JAVA_HOME"] = _jdk
        os.environ["PATH"] = _jdk + r"\bin;" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel

MODEL_PATH = os.path.abspath("models/dt_pipeline").replace("\\", "/")

# Scheduled days per shipping mode (from the dataset analysis)
SHIPPING_MODES = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}

CAT = ["shipping_mode", "order_region", "customer_segment", "market",
       "type", "department_name", "category_name"]
NUM = ["days_for_shipment_scheduled", "order_item_quantity", "order_item_discount_rate",
       "order_item_product_price", "order_month", "order_weekday", "order_hour"]


def get_spark():
    return (SparkSession.builder.appName("SupplyChainRecommender").master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def load_model():
    return PipelineModel.load(MODEL_PATH)


def score_modes(spark, model, order):
    """Late-delivery probability for this order under every shipping mode."""
    schema = StructType([StructField(c, StringType(), True) for c in CAT] +
                        [StructField(c, DoubleType(), True) for c in NUM])
    rows = []
    for mode, days in SHIPPING_MODES.items():
        o = dict(order)
        o["shipping_mode"] = mode
        o["days_for_shipment_scheduled"] = float(days)
        rows.append(tuple(o[c] if c in CAT else float(o[c]) for c in CAT + NUM))
    out = model.transform(spark.createDataFrame(rows, schema)).select("shipping_mode", "probability").collect()
    return {r["shipping_mode"]: float(r["probability"][1]) for r in out}


def recommend(risks, chosen_mode):
    best = min(risks, key=risks.get)
    chosen, best_risk = risks[chosen_mode], risks[best]
    if best == chosen_mode or abs(chosen - best_risk) < 1e-9:
        msg = f"**{chosen_mode}** already has the lowest predicted late-delivery risk ({chosen:.0%}). Keep it."
    else:
        msg = (f"Switch from **{chosen_mode}** to **{best}**: predicted late-delivery risk "
               f"drops from {chosen:.0%} to {best_risk:.0%} ({(chosen - best_risk) * 100:.0f} points lower).")
    return best, msg