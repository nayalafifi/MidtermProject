import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Create a sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")


df = pd.read_csv('mmh.csv')
df.head()
st.markdown("## Columns")
df.columns

## Description of Dataset

num = st.number_input('No of Rows',5,10)
st.dataframe(df.head(num))

### Description of the dataset

st.dataframe(df.describe())

if st.button("Show Describe Code"):
        code = '''df.describe()'''
        st.code(code, language='python')

if st.button("Generate Report"):
  import streamlit as st
  import streamlit.components.v1 as components



st.markdown("## Visualization")

tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])

tab1.subheader("Line Chart")
# Display a line chart for the selected variables
tab1.line_chart(data=df, x="Age", y="Music effects", width=0, height=0, use_container_width=True)

tab2.subheader("Bar Chart")
# Display a bar chart for the selected variables
tab2.bar_chart(data=df, x="Age", y="Music effects", use_container_width=True)

fig, ax = plt.subplots()
sns.distplot(df['Hours per day'], ax=ax)
st.pyplot(fig)



st.markdown("## Linear Regression")


sampled_df = df.sample(n=400)
sampled_df_10_columns = sampled_df.iloc[:, :10]
sns.pairplot(sampled_df_10_columns)

