import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home Page", "Pre-Study Assumptions", "General Graphs","Linear regression Results","Conclusions and Results","Evaluation"])

if selection == "Home Page":
        st.title("Mental Health and Music Correlation Data Visualisation and Predictions!")
        st.write("Check out the navigations sidebar to be directed to your desired page")
        st.image("mental.jpg")
        st.header("Components of This Study....")
        st.write("This data set is the result of a survey conducted in 2020. It mainly touches on people music listening habits and its effects on mental health. The key features of this data set include overall efffects of music, age, streaming platform, hours per day, favourite Genre and more. This dataset is an optimal way to provide insights into people's music listening patterns, preferences, and the effects on mental health among different age groups and lifestyles. ​This study will consist of three main parts: an introduction and Brief into the data set, some visuals/graphs to visualise the data set, and the predictions we have compared to the predictions we accessed through linear regression. We mainly focused our study and prediction prior to the study on mental disorder levels in music listeners for several reasons, 1- our data was very suitable for this specifc study because the study sample associated a numerical value for their mental disorder. 2- Additionally, since linear regression models only work with one variable, we decided to allow the user to choose this variable, and view the actual vs. predicted graph accordingly.")
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
        
        st.image("image2.jpg")
        
        st.markdown("## Visualization")
        
        tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])
        
        tab1.subheader("Line Chart")
        # Display a line chart for the selected variables
        tab1.line_chart(data=df, x="Age", y="Music effects", width=0, height=0, use_container_width=True)
        
        tab2.subheader("Bar Chart")
        # Display a bar chart for the selected variables
        tab2.bar_chart(data=df, x="Age", y="Music effects", use_container_width=True)


elif selection == "Pre-Study Assumptions":
    st.write("# Welcome to the Pre-Study Assumptions!")
    st.image("assumptions.jpg")

elif selection == "General Graphs":
        st.write("# Welcome to the General Graphs page!")
        df = pd.read_csv('CLEANmmh.csv')
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

elif selection == "Linear regression Results":
        st.write("# Welcome to the linear regression page! ")
        df = pd.read_csv('CLEANmmh.csv')
        quantitative_df = df.select_dtypes(include=[np.number])
        # quantitative_df = df.select_dtypes(include=[np.number])
        
        
        # Use 'st.selectbox' to create a dropdown menu
        selection = st.selectbox(
            "Click below to change the variable of the linear regression:",
            quantitative_df[["Anxiety", "Depression", "OCD", "Insomnia"]].columns
        )
        
        X = quantitative_df.drop(selection, axis=1)
        y = quantitative_df[selection]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_train,y_train)
        # coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
        # coeff_df
        df.drop("BPM", axis=1)
        
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
        
        
        MAE = metrics.mean_absolute_error(y_test, pred)
        MSE = metrics.mean_squared_error(y_test, pred)
        RMSE = np.sqrt(MSE)
        
        st.write(f'MAE: {MAE}')
        st.write(f'MSE: {MSE}')
        st.write(f'RMSE: {RMSE}')
        
elif selection == "Conclusions and Results":
        st.write("# Welcome to the conclusions and results page! ")
        st.header("Below is our conclusions")
        st.write("After looking through the graph visualizations, and studying the linear regression models and predictions, we found that firstly, the majority of music listeners expressed an improvement in their mental health. Next, we also found that listeners of younger ages generally have higher levels of the specific mental health disorder, and this applies to Anxiety, Insomnia, OCD, and Depression.")
        st.image("pasted.png")
        st.header("Furthermore...")
        st.write("The Linear regression models indicated to us that for instance when the user selects anxiety as the test disorder, the model's predictions are approximately a Mean Absolute Error of  1.688 units off from the actual anxiety levels. The Mean Squared Error is approximately 4.266 indicating the the difference between the predicted value and the actual value squared, so if its a large margin of error the squared value would be very large, and if small than vice versa. The Root Mean Squared Error  is  2.065, the square root of the MSE. While the error margins are not too large, there still is room for improvement and we hope that as we advance through the semester we can better learn how to limit these errors and improve this project.")
        st.image("pasted1.png")


elif selection == "Evaluation":
        st.write("# Please scan the QR code below to give us your review of our app!")
        st.image("qr.png")


