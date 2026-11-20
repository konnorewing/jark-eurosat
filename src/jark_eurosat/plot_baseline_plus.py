import numpy as np
import pandas as pd
from great_tables import GT
import plotly.express as px


baseline_accuracy = pd.read_csv('../../results/baseline_accuracy.csv').iloc[0]
baseline_plus_accuracy = pd.read_csv('../../results/baseline_plus_accuracy.csv').iloc[0]

accuracy_table = pd.DataFrame({
  "Baseline": baseline_accuracy,
  "Baseline Plus": baseline_plus_accuracy
})
accuracy_table["Change [%]"] = accuracy_table["Baseline Plus"] - accuracy_table["Baseline"]

gt_table = (
  GT(accuracy_table.reset_index(names="Class"))
  .fmt_number(columns=["Baseline", "Baseline Plus"], decimals=2)
  .fmt_number(columns=["Change [%]"], decimals=2, force_sign=True)
)
gt_table.gtsave("../../results/baseline_plus_accuracy_table.png")

# Bar Chart
overall_accuracies = accuracy_table.loc["overall", ["Baseline", "Baseline Plus"]]
fig = px.bar(
  x=overall_accuracies.index,
  y=overall_accuracies.values,
  color=overall_accuracies.index,
  labels={"x": "Model", "y": "Validation Accuracy [%]"},
  title="Overall Validation Accuracy"
)
fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
fig.update_layout(showlegend=False)
fig.write_image("../../results/baseline_plus_accuracy_bar_chart.png")