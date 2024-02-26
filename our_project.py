import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import looker as lk

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

