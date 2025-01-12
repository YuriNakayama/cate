import pandas as pd
import streamlit as st

df = pd.read_csv(
    r"/Users/yurinakayama/data/cate/output/datasets_stats/base_stats.csv", index_col=0
)

# Create a Streamlit dashboard
st.title("Dataset Statistics Dashboard")
st.write("This dashboard displays statistics from the dataset.")

# Display the dataframe
st.write("Dataframe:")
st.dataframe(df.T)

# Display line chart of the dataframe
evaluation_indices = [""]
st.write("Line Chart:")
st.line_chart(df.T)
