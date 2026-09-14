import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import Retriever, escalation_reasons
from evaluate import route_metrics

# Load data
golden = pd.read_csv("golden_set.csv")
threads = pd.read_csv("data/amazonhelp_threads.csv")
intent_preds = pd.read_csv("results/intent_preds.csv")

# Initialize retriever
train_threads = threads[~threads.conv_id.isin(golden.conv_id)]
outbounds = train_threads[~train_threads.inbound]
train_conv_ids = outbounds.conv_id.unique()
train_pairs = []
for cid in train_conv_ids:
    c = train_threads[train_threads.conv_id == cid].sort_values("depth")
    in_msgs = c[c.inbound]
    out_msgs = c[~c.inbound]
    if len(in_msgs) and len(out_msgs):
        train_pairs.append((in_msgs.iloc[-1].text, out_msgs.iloc[0].text))

retriever = Retriever([p[0] for p in train_pairs], [p[1] for p in train_pairs])

# Run improved routing policy
new_routes = []
for i, row in golden.iterrows():
    t = row.text
    intent = intent_preds.loc[i, "llm"]
    conf = intent_preds.loc[i, "llm_conf"]
    max_sim = retriever.max_sim(t)
    
    reasons = escalation_reasons(t, intent, conf, max_sim)
    new_routes.append("human" if reasons else "auto")

# Get original metrics from summary.json
with open("results/summary.json") as f:
    summary = json.load(f)
    
old_m = summary["route"]["llm"]

# Calculate new metrics
new_m = route_metrics(golden.route, new_routes)

# Create comparison dataframe
comparison = pd.DataFrame([
    {
        "system": "Original policy",
        "route_accuracy": old_m["route_accuracy"],
        "missed_escalations": old_m["missed_escalations"],
        "false_alarms": old_m["false_alarms"],
        "escalation_rate": old_m["escalation_rate"]
    },
    {
        "system": "Improved policy (Capability/Risk Layer)",
        "route_accuracy": new_m["route_accuracy"],
        "missed_escalations": new_m["missed_escalations"],
        "false_alarms": new_m["false_alarms"],
        "escalation_rate": new_m["escalation_rate"]
    }
])

comparison.to_csv("results/route_comparison.csv", index=False)
print(comparison)
