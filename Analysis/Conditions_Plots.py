import numpy as np
import sqlite3 as lite
import pandas as pd
import plotly.express as px

with lite.connect("/home/jedediah/Downloads/Conditions.db") as conn:
    df = pd.read_sql_query("SELECT * FROM conditions", conn)

dfg = df[df.Humidity < 100]  # To filter out random humidity errors in sensor

fig = px.scatter(
    dfg,
    x="Local",
    y="Temp",
    color="Humidity",
    template="plotly_dark",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig.show()

fig2 = px.scatter(
    dfg,
    x="Local",
    y="Humidity",
    color="Humidity",
    template="plotly_dark",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig2.show()
