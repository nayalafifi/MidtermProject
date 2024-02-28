import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn import metrics as mt



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

# Create a pairplot
sns_plot = sns.pairplot(sampled_df_10_columns)

# Show the plot in Streamlit
st.pyplot(sns_plot)

plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size
sns.histplot(df['Age'], kde=True)  # kde=True adds a density curve

# Show the plot in Streamlit
st.pyplot(plt)



st.markdown("## Linear Regression")

quantitative_df = df.select_dtypes(include=[np.number])
# quantitative_df = df.select_dtypes(include=[np.number])

# Use 'st.selectbox' to create a dropdown menu
selection = st.selectbox(
    "Select the disorder you would like to predict",
    quantitative_df[["Anxiety", "Depression", "OCD", "Insomnia"]].columns
)

X = quantitative_df.drop(selection, axis=1)
y = quantitative_df[selection]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





