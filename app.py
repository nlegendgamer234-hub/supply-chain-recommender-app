import datetime, json, os
import pandas as pd
import streamlit as st

SHIPPING_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]

st.set_page_config(page_title="Supply Chain Recommender", page_icon="📦", layout="wide")
st.title("📦 Supply Chain Delivery Recommendation System")
st.caption("PySpark MLlib Decision Tree Classifier · DataCo Smart Supply Chain data (180,519 orders)")


@st.cache_resource(show_spinner="Starting the prediction engine...")
def load_engine():
    """Live Spark if available (laptop); otherwise the exported table (free web hosting)."""
    if os.environ.get("FORCE_LITE") != "1":
        try:
            import recommend as rec
            spark = rec.get_spark()
            spark.sparkContext.setLogLevel("ERROR")
            model = rec.load_model()
            return {"mode": "spark", "rec": rec, "spark": spark, "model": model}
        except Exception:
            pass
    with open("risk_table.json", encoding="utf-8") as f:
        return {"mode": "lite", "table": json.load(f)}


@st.cache_data
def get_options():
    with open("options.json", encoding="utf-8") as f:
        return json.load(f)


def get_risks(engine, order, hour):
    if engine["mode"] == "spark":
        return engine["rec"].score_modes(engine["spark"], engine["model"], order)
    row = engine["table"][str(int(hour))]
    return {m: float(row[m]) for m in SHIPPING_MODES}


def make_recommendation(risks, chosen_mode):
    best = min(risks, key=risks.get)
    chosen, best_risk = risks[chosen_mode], risks[best]
    if best == chosen_mode or abs(chosen - best_risk) < 1e-9:
        msg = f"**{chosen_mode}** already has the lowest predicted late-delivery risk ({chosen:.0%}). Keep it."
    else:
        msg = (f"Switch from **{chosen_mode}** to **{best}**: predicted late-delivery risk "
               f"drops from {chosen:.0%} to {best_risk:.0%} ({(chosen - best_risk) * 100:.0f} points lower).")
    return best, msg


engine = load_engine()
opts = get_options()

if engine["mode"] == "spark":
    st.caption("⚙️ Engine: live Apache Spark (PySpark MLlib PipelineModel)")
else:
    st.caption("⚙️ Engine: predictions exported from the Spark-trained model "
               "(lightweight online mode; training and analysis were done in PySpark)")

tab1, tab2, tab3 = st.tabs(["🎯 Recommend", "📊 Data insights", "ℹ️ About"])

# ---------------- Tab 1: recommendation ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        chosen = st.selectbox("Shipping mode chosen by customer", SHIPPING_MODES)
        region = st.selectbox("Order region", opts["order_region"],
                              index=opts["order_region"].index("Western Europe") if "Western Europe" in opts["order_region"] else 0)
        market = st.selectbox("Market", opts["market"])
        segment = st.selectbox("Customer segment", opts["customer_segment"])
    with c2:
        ptype = st.selectbox("Payment type", opts["type"])
        dept = st.selectbox("Department", opts["department_name"])
        cat = st.selectbox("Product category", opts["category_name"])
    with c3:
        qty = st.number_input("Quantity", min_value=1, max_value=20, value=1)
        discount = st.slider("Discount rate", 0.0, 0.25, 0.05, 0.01)
        price = st.number_input("Unit price", min_value=1.0, value=100.0)
        d = st.date_input("Order date", datetime.date.today())
        t = st.time_input("Order time", datetime.time(12, 0))

    if st.button("Get recommendation", type="primary"):
        order = {
            "order_region": region, "customer_segment": segment, "market": market,
            "type": ptype, "department_name": dept, "category_name": cat,
            "order_item_quantity": float(qty), "order_item_discount_rate": float(discount),
            "order_item_product_price": float(price), "order_month": float(d.month),
            "order_weekday": float((d.weekday() + 1) % 7 + 1),  # Spark: 1 = Sunday
            "order_hour": float(t.hour),
        }
        with st.spinner("Scoring..."):
            risks = get_risks(engine, order, t.hour)
        best, msg = make_recommendation(risks, chosen)

        m1, m2 = st.columns(2)
        m1.metric(f"Late risk · {chosen} (chosen)", f"{risks[chosen]:.0%}")
        delta = None if best == chosen else f"{(risks[best] - risks[chosen]) * 100:.0f} pts"
        m2.metric(f"Late risk · {best} (recommended)", f"{risks[best]:.0%}",
                  delta=delta, delta_color="inverse")
        (st.success if best == chosen else st.warning)(msg)

        st.subheader("Predicted late-delivery risk by shipping mode")
        st.bar_chart(pd.Series(risks, name="Late-delivery probability"))
        st.caption("In this dataset, faster modes promise tighter delivery windows (Same Day = 0 days, "
                   "First Class = 1, Second = 2, Standard = 4), so they miss their promise more often.")

# ---------------- Tab 2: insights ----------------
with tab2:
    st.subheader("Model performance (held-out 20% test set)")
    a, b, c, d_ = st.columns(4)
    a.metric("Accuracy", "71.96%")
    b.metric("F1 score", "0.716")
    c.metric("AUC", "0.707")
    d_.metric("Majority baseline", "≈55%")
    st.write("The decision tree (max depth 3) relies almost entirely on **shipping mode** and **order hour**. "
             "Region, segment and product columns carried no signal in this dataset.")

    st.subheader("Exploratory analysis (Spark SQL)")
    for f, cap in [("late_by_shipping_mode.png", "Late delivery % by shipping mode"),
                   ("late_by_region.png", "Late delivery % by region"),
                   ("scheduled_vs_real_days.png", "Scheduled vs real shipping days")]:
        if os.path.exists("charts/" + f):
            st.image("charts/" + f, caption=cap, width=650)

    if os.path.exists("models/tree_rules.txt"):
        with st.expander("Learned decision tree rules"):
            st.markdown("""
**In plain English** (`feature 0` = shipping mode, `feature 13` = order hour):
- **Standard Class** → predicted **on time**
- **First Class / Second Class** → predicted **late**
- **Same Day** → predicted **on time if ordered at or before 11:00, otherwise late**
""")
            st.code(open("models/tree_rules.txt", encoding="utf-8").read())

# ---------------- Tab 3: about ----------------
with tab3:
    st.markdown("""
**Pipeline:** CSV → Spark DataFrame → cleaning and feature engineering (StringIndexer, VectorAssembler)
→ MLlib `DecisionTreeClassifier` → saved `PipelineModel` → Streamlit app.

**Target:** `late_delivery_risk` (1 = late). **Leakage avoided:** `delivery_status` and
`days_for_shipping_real` are excluded because they are only known after delivery.

**Recommendation logic:** the model scores the same order under each shipping mode and recommends the
mode with the lowest predicted late-delivery probability.

**Deployment:** on a laptop the app scores orders with live Spark. The free web version serves the same
Spark model's predictions from an exported lookup table, because free hosting cannot run Spark.

**Tools:** Python, Apache Spark / PySpark (MLlib, Spark SQL), Streamlit, pandas, matplotlib.
""")