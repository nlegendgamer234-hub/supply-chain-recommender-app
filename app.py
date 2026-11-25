import datetime, json, os
import streamlit as st

SHIPPING_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
PROMISE = {"Standard Class": "promises 4 days", "Second Class": "promises 2 days",
           "First Class": "promises 1 day", "Same Day": "promises same day"}

st.set_page_config(page_title="Delivery Risk Advisor", page_icon="📦", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;800&display=swap');
.stMarkdown, .stApp h1, .stApp h2, .stApp h3 {font-family: 'Manrope', 'Segoe UI', sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
#MainMenu, footer {visibility: hidden;}

.hero {background: #0f3d5e; color: #fff; padding: 30px 34px 22px 34px; border-radius: 6px 26px 6px 26px; margin-bottom: 22px;}
.hero-title {font-size: 2.1rem; font-weight: 800; line-height: 1.2; margin-bottom: 8px;}
.hero-sub {font-size: 1.02rem; opacity: .9; max-width: 720px; margin-bottom: 16px; line-height: 1.5;}
.pill {display: inline-block; border: 1px solid rgba(255,255,255,.45); color: #fff; padding: 4px 13px; border-radius: 999px; font-size: .82rem; margin: 0 8px 6px 0;}
.pill.live {background: #f2b134; border-color: #f2b134; color: #1a1a1a; font-weight: 600;}

.panel-title {font-weight: 800; font-size: 1.05rem; margin: 4px 0 2px 0;}
.hint {opacity: .7; font-size: .9rem; margin-bottom: 8px;}

.empty {border: 2px dashed rgba(128,128,128,.4); border-radius: 14px; padding: 34px 26px; text-align: center; line-height: 1.6;}
.empty b {font-size: 1.1rem;}

.reco {border-radius: 8px 22px 8px 22px; padding: 18px 24px; margin-bottom: 14px; border-left: 8px solid;}
.reco.keep {background: rgba(46,158,91,.13); border-color: #2e9e5b;}
.reco.switch {background: rgba(242,177,52,.18); border-color: #f2b134;}
.reco .t {font-size: 1.5rem; font-weight: 800; margin-bottom: 4px;}
.reco .s {opacity: .92; line-height: 1.5;}
.reco .c {margin-top: 8px; font-size: .9rem; opacity: .85;}

.lane {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 8px 18px; margin-bottom: 12px;}
.bar-row {display: flex; align-items: center; gap: 14px; margin: 14px 0;}
.bar-label {width: 175px; font-weight: 700; font-size: .96rem; line-height: 1.25;}
.bar-label small {display: block; font-weight: 400; opacity: .65; font-size: .76rem;}
.bar-track {flex: 1; background: rgba(128,128,128,.2); border-radius: 999px; height: 20px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.bar-val {width: 54px; text-align: right; font-weight: 800; font-size: 1.05rem;}
.tag {font-size: .68rem; padding: 2px 8px; border-radius: 999px; margin-left: 6px; color: #fff; vertical-align: middle; font-weight: 600;}
.tag.chosen {background: #64748b;}
.tag.best {background: #2e9e5b;}

.kpi {border: 1px solid rgba(128,128,128,.3); border-radius: 12px; padding: 16px 12px; text-align: center; margin-bottom: 10px;}
.kpi .v {font-size: 2rem; font-weight: 800; line-height: 1.15;}
.kpi .l {font-size: .85rem; opacity: .72; margin-top: 2px;}

.cm {border-collapse: collapse; width: 100%; text-align: center; margin: 6px 0 4px 0;}
.cm th {font-weight: 600; font-size: .82rem; opacity: .75; padding: 6px;}
.cm td {padding: 14px 8px; font-weight: 800; font-size: 1.2rem; border: 3px solid transparent;}
.cm td.good {background: rgba(46,158,91,.2);}
.cm td.bad {background: rgba(214,69,69,.18);}
.cm td small {display: block; font-weight: 400; opacity: .7; font-size: .72rem;}

.flow {display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 14px 0;}
.step {flex: 1 1 150px; border: 1px solid rgba(128,128,128,.3); border-top: 5px solid #1f7a8c; border-radius: 4px 14px 4px 14px; padding: 12px 14px;}
.step b {display: block; margin-bottom: 4px;}
.step span {font-size: .86rem; opacity: .8; line-height: 1.4;}
.note {border-left: 5px solid #1f7a8c; background: rgba(31,122,140,.1); padding: 10px 16px; border-radius: 0 10px 10px 0; margin: 8px 0 12px 0; line-height: 1.5;}
.foot {opacity: .6; font-size: .82rem; margin-top: 26px; text-align: center;}

@media (max-width: 700px) {
  .hero-title {font-size: 1.5rem;}
  .bar-row {flex-wrap: wrap;}
  .bar-label {width: 100%;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


# ------------------------------------------------------------------ engine
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


def risk_color(p):
    return "#2e9e5b" if p < 0.45 else ("#f2b134" if p < 0.75 else "#d64545")


engine = load_engine()
opts = get_options()
live = engine["mode"] == "spark"

html(f"""
<div class="hero">
<div class="hero-title">Which shipping mode will arrive on time?</div>
<div class="hero-sub">Supply Chain Delivery Recommendation System. A decision tree trained with PySpark on
180,519 real orders checks the late-delivery risk of every shipping option and points to the safest one.</div>
<span class="pill">PySpark MLlib</span>
<span class="pill">72% accuracy on unseen orders</span>
<span class="pill live">{"Running live on Apache Spark" if live else "Serving predictions from the Spark model"}</span>
</div>
""")

tab1, tab2, tab3 = st.tabs(["Check an order", "Model results", "How it works"])


# ------------------------------------------------------------------ tab 1
def preset(mode, hour):
    st.session_state["chosen"] = mode
    st.session_state["otime"] = datetime.time(hour, 0)
    st.session_state["autorun"] = True


st.session_state.setdefault("chosen", "Standard Class")
st.session_state.setdefault("otime", datetime.time(12, 0))


def render_result(res):
    risks, chosen = res["risks"], res["chosen"]
    best = min(risks, key=risks.get)
    cp, bp = risks[chosen], risks[best]
    if best == chosen or abs(cp - bp) < 1e-9:
        cls, title = "keep", f"Keep {chosen}"
        sub = f"It already has the lowest late-delivery risk of the four options ({cp:.0%})."
    else:
        cls, title = "switch", f"Switch to {best}"
        sub = (f"The risk of a late delivery falls from {cp:.0%} with {chosen} to {bp:.0%} with {best}, "
               f"which is {abs(cp - bp) * 100:.0f} points lower.")
    caution = ""
    if bp >= 0.5:
        caution = '<div class="c">Even the safest option is above 50%, so tell the customer a delay is possible.</div>'
    html(f'<div class="reco {cls}"><div class="t">{title}</div><div class="s">{sub}</div>{caution}</div>')

    rows = ""
    for m in sorted(risks, key=risks.get):
        p = risks[m]
        tags = ""
        if m == chosen:
            tags += '<span class="tag chosen">chosen</span>'
        if m == best:
            tags += '<span class="tag best">safest</span>'
        rows += (f'<div class="bar-row"><div class="bar-label">{m}{tags}<small>{PROMISE[m]}</small></div>'
                 f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.0f}%;background:{risk_color(p)}"></div></div>'
                 f'<div class="bar-val">{p:.0%}</div></div>')
    html('<div class="panel-title">Late-delivery risk by shipping mode</div>')
    html(f'<div class="lane">{rows}</div>')
    st.caption(f"Order checked: {res['summary']}")

    with st.expander("Why does the model say this?"):
        st.markdown(
            "The decision tree learned three simple rules from the 180,519 orders:\n\n"
            "- **Standard Class** orders usually arrive on time.\n"
            "- **First Class** and **Second Class** orders are usually late.\n"
            "- **Same Day** orders arrive on time only when they are placed by 11:00.\n\n"
            "Faster modes promise tighter delivery windows (Same Day 0 days, First Class 1, Second Class 2, "
            "Standard Class 4), so they miss their promise more often in this data.")


with tab1:
    left, right = st.columns([5, 6], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Try an example</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            b1.button("First Class at 9 AM", key="p0", on_click=preset, args=("First Class", 9))
            b2.button("Same Day at 9 AM", key="p1", on_click=preset, args=("Same Day", 9))
            b3.button("Same Day at 3 PM", key="p2", on_click=preset, args=("Same Day", 15))

            st.markdown('<div class="panel-title">The order</div>', unsafe_allow_html=True)
            chosen = st.selectbox("Shipping mode the customer picked", SHIPPING_MODES, key="chosen")
            region = st.selectbox("Order region", opts["order_region"],
                                  index=opts["order_region"].index("Western Europe") if "Western Europe" in opts["order_region"] else 0)
            m1, m2 = st.columns(2)
            market = m1.selectbox("Market", opts["market"])
            segment = m2.selectbox("Customer segment", opts["customer_segment"])

            st.markdown('<div class="panel-title">Product and payment</div>', unsafe_allow_html=True)
            ptype = st.selectbox("Payment type", opts["type"])
            p1, p2 = st.columns(2)
            dept = p1.selectbox("Department", opts["department_name"])
            cat = p2.selectbox("Product category", opts["category_name"])

            st.markdown('<div class="panel-title">Quantity, price and time</div>', unsafe_allow_html=True)
            q1, q2 = st.columns(2)
            qty = q1.number_input("Quantity", min_value=1, max_value=20, value=1)
            price = q2.number_input("Unit price", min_value=1.0, value=100.0)
            discount = st.slider("Discount rate", 0.0, 0.25, 0.05, 0.01)
            d1, d2 = st.columns(2)
            d = d1.date_input("Order date", datetime.date.today())
            t = d2.time_input("Order time", key="otime")

            go = st.button("Check delivery risk", type="primary", key="check")

    if go or st.session_state.pop("autorun", False):
        order = {
            "order_region": region, "customer_segment": segment, "market": market,
            "type": ptype, "department_name": dept, "category_name": cat,
            "order_item_quantity": float(qty), "order_item_discount_rate": float(discount),
            "order_item_product_price": float(price), "order_month": float(d.month),
            "order_weekday": float((d.weekday() + 1) % 7 + 1),  # Spark: 1 = Sunday
            "order_hour": float(t.hour),
        }
        with st.spinner("Scoring the order..."):
            risks = get_risks(engine, order, t.hour)
        st.session_state["result"] = {
            "risks": risks, "chosen": chosen,
            "summary": f"{chosen}, {region}, {t.strftime('%H:%M')}",
        }

    with right:
        if "result" in st.session_state:
            render_result(st.session_state["result"])
        else:
            html("""
            <div class="empty"><b>No order checked yet</b><br>
            Pick an example on the left or fill in an order, then select <b>Check delivery risk</b>.
            You will see the late-delivery risk of all four shipping modes and the safest choice.</div>
            """)

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("### How well does the model work?")
    st.write("These numbers come from 35,872 orders the model never saw during training (a held-out 20% test set).")
    k1, k2, k3, k4 = st.columns(4)
    for col, v, l in [(k1, "71.96%", "Accuracy"), (k2, "0.716", "F1 score"),
                      (k3, "0.707", "AUC"), (k4, "55%", "Accuracy of always guessing late")]:
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([5, 6], gap="large")
    with c1:
        st.markdown("**Right and wrong calls**")
        html("""
        <table class="cm">
        <tr><th></th><th>Model said on time</th><th>Model said late</th></tr>
        <tr><th>Really on time</th><td class="good">14,359<small>correct</small></td><td class="bad">1,924<small>false alarm</small></td></tr>
        <tr><th>Really late</th><td class="bad">8,136<small>missed</small></td><td class="good">11,453<small>correct</small></td></tr>
        </table>
        """)
    with c2:
        st.markdown("**What this means**")
        st.markdown(
            "- When the model says an order will be late, it is right **86%** of the time.\n"
            "- It catches **58%** of the orders that really arrive late.\n"
            "- It correctly clears **88%** of the orders that arrive on time.\n"
            "- It beats always guessing the most common answer by about 17 points.")

    st.markdown("### What the data shows (Spark SQL)")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        if os.path.exists("charts/late_by_shipping_mode.png"):
            st.image("charts/late_by_shipping_mode.png", caption="Share of late deliveries by shipping mode")
    with g2:
        if os.path.exists("charts/scheduled_vs_real_days.png"):
            st.image("charts/scheduled_vs_real_days.png", caption="Days promised against days taken")
    g3, g4 = st.columns(2, gap="large")
    with g3:
        if os.path.exists("charts/late_by_region.png"):
            st.image("charts/late_by_region.png", caption="Late deliveries are about 55% in almost every region")
    with g4:
        html("""
        <div class="note"><b>What this tells us.</b> Shipping mode explains almost everything. Region, customer
        segment and product barely change the late rate, so the tree ignores them.</div>
        """)
        if os.path.exists("models/tree_rules.txt"):
            with st.expander("See the learned tree rules"):
                st.markdown("`feature 0` is shipping mode and `feature 13` is order hour. "
                            "Standard Class is predicted on time, First and Second Class are predicted late, "
                            "and Same Day is predicted on time up to 11:00.")
                st.code(open("models/tree_rules.txt", encoding="utf-8").read())

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("### From raw orders to a recommendation")
    html("""
    <div class="flow">
    <div class="step"><b>Load</b><span>180,519 orders read from CSV into a Spark DataFrame.</span></div>
    <div class="step"><b>Clean</b><span>Missing values dropped, order date split into month, weekday and hour.</span></div>
    <div class="step"><b>Encode</b><span>StringIndexer and VectorAssembler turn columns into model features.</span></div>
    <div class="step"><b>Train</b><span>MLlib DecisionTreeClassifier learns which orders arrive late.</span></div>
    <div class="step"><b>Recommend</b><span>The same order is scored under all four modes and the lowest risk wins.</span></div>
    </div>
    """)

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**What the model predicts**")
        st.write("The target is `late_delivery_risk`, where 1 means the order arrived late. "
                 "`delivery_status` and `days_for_shipping_real` are left out on purpose, because they are only "
                 "known after delivery and would let the model cheat.")
        st.markdown("**Technology**")
        st.write("Python, Apache Spark and PySpark (MLlib and Spark SQL), Streamlit, pandas and matplotlib.")
    with b:
        st.markdown("**Limits to keep in mind**")
        st.markdown(
            "- The model relies almost only on shipping mode and order hour. Other details do not change the answer.\n"
            "- It learns from one company's past orders, so it shows patterns in this data, not a promise about any real carrier.\n"
            "- The recommendation favours slower modes because this dataset promises faster modes tighter deadlines.")
        st.markdown("**About this web version**")
        st.write("On a laptop the app scores orders with live Spark. The free web version serves predictions "
                 "exported from the same Spark model, because free hosting cannot run Spark.")

html('<div class="foot">Big Data Tools and Technologies project: Recommendation System in Supply Chain Data using a Decision Tree Classifier</div>')
