import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
import random
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


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

