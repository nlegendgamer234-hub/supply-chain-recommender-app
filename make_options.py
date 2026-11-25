import json, re
import pandas as pd

clean = lambda c: re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()
cols = ["shipping_mode", "order_region", "customer_segment", "market",
        "type", "department_name", "category_name"]

df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1",
                 usecols=lambda c: clean(c) in cols)
df.columns = [clean(c) for c in df.columns]
opts = {c: sorted(df[c].dropna().unique().tolist()) for c in cols}

with open("options.json", "w", encoding="utf-8") as f:
    json.dump(opts, f, indent=1)
print("options.json created")