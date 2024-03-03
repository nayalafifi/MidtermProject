import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt



# Create a sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")


df = pd.read_csv('CLEANmmh.csv')
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



st.markdown("## General graphs")


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


quantitative_df = df.select_dtypes(include=[np.number])
df_sample_q = quantitative_df.sample(n=400).reset_index(drop=True)






# pivot_table = pd.pivot_table(df, index='Music effects', columns='Hours per day', aggfunc='size', fill_value=0)

# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_table, annot=True, fmt="d")  # Use fmt="d" to format numbers as integers
# plt.show()

pivot_table = pd.pivot_table(df, index='Music effects', columns='Hours per day', aggfunc='size', fill_value=0)

# Display the pivot table in your Streamlit app
st.write(pivot_table)

# Create a heatmap using seaborn and display it using Streamlit
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt="d")  # Use fmt="d" to format numbers as integers

# Display the plot in Streamlit
st.pyplot(plt)

df.drop("BPM", axis=1)



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

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
# coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
# coeff_df

X.columns #checking if the selection works


feature_names = [f'Feature_{i}' for i in list(X.columns)]
df_X = pd.DataFrame(X, columns=feature_names)
# Coefficients represent the importance in linear regression
coefficients = lin_reg.coef_

# Making the coefficients positive to compare magnitude
importance = np.abs(coefficients)

# Plotting feature importance with feature names
feature_names = [f'Feature_{i}' for i in list(X.columns)]
df_X = pd.DataFrame(X, columns=feature_names)
coefficients = lin_reg.coef_

importance = np.abs(coefficients)

# Plotting feature importance with feature names
fig, ax = plt.subplots(figsize=(10, 8))  # Use Streamlit's pyplot instead of plt.show()
ax.barh(feature_names, importance)
ax.set_xlabel('Absolute Coefficient Value')
ax.set_title('Feature Importance (Linear Regression)')

# Streamlit uses st.pyplot() to display matplotlib figures
st.pyplot(fig)

pred = lin_reg.predict(X_test)
# Plotting
plt.figure(figsize=(10,7))
plt.figure(figsize=(10,7))
plt.title(f"Actual vs. Predicted Levels of {selection} in Music Listeners", fontsize=20)
plt.xlabel(f"Actual Levels of {selection} in Music Listeners", fontsize=16)
plt.ylabel(f"Predicted Levels of {selection} in Music Listeners", fontsize=16)
plt.scatter(x=y_test, y=pred)

# Use Streamlit to render the plot
st.pyplot(plt)
