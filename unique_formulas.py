import json
import os
import pandas as pd

metadata = json.load(open("data/oc22/is2re-total/metadata.json"))
entries = {k: v for k, v in metadata.items() if v.get("ads_symbols", "") != ""}

os.makedirs("output", exist_ok=True)

ads = sorted(set(v["ads_symbols"] for v in entries.values()))
bulk = sorted(set(v["bulk_symbols"] for v in entries.values()))

pd.DataFrame({"ads_symbols": ads}).to_csv("output/unique_adsorbates.csv", index=False)
pd.DataFrame({"bulk_symbols": bulk}).to_csv("output/unique_adsorbents.csv", index=False)

print(f"Unique adsorbates: {len(ads)} → output/unique_adsorbates.csv")
print(f"Unique adsorbents: {len(bulk)} → output/unique_adsorbents.csv")
