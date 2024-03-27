# Importing the Required Libraries & Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
data = sns.load_dataset("diamonds")

#Page Configuration
st.set_page_config(page_title="Diamond Price Prediction",layout="wide",page_icon="💎")
st.title("Diamond Price Prediction")
st.image("Diamond.png",width=350)


#Sidebar of the WebApp
menu=st.sidebar.radio("Menu",["Home","Explore Diamond's Features"])

if menu=="Home":
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # One-hot encode the 'cut' column
    encoded_data = pd.get_dummies(data, columns=['cut'], drop_first=True)

    # Select features and target
    X = encoded_data[['carat']]
    y = encoded_data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    st.write("#### Diamond Insights: Predicting Prices for Smart Investments...")
    value = st.number_input("Carat", 0.20, 5.01, step=0.01)
    prediction = lr.predict([[value]])

    if st.button("Predict Price($)"):
        st.write(f"Predicted Price: ${prediction[0]:.2f}")
    
    
if menu=="Explore Diamond's Features":
    st.write("Shape of the Dataset",data.shape) 
    st.header("Tabular Data of the Diamond")
    if st.checkbox("Sample Data"):
        st.table(data.head(20))
    if st.checkbox("Data Statistics"):
        st.table(data.describe())
    if st.checkbox("Correlation Graph"):
          numeric_data = data.select_dtypes(include=np.number)
          fig, ax = plt.subplots(figsize=(8, 2.5))
          sns.heatmap(numeric_data.corr(), annot=True)
          st.pyplot(fig)
        
    st.header("Graphical Representation") 
    graph=st.selectbox("Select a Graph",options=[None,"Scatter Plot","Bar Graph","Histogram"]) 
    if graph=="Scatter Plot":  
        value=st.slider("Filter Data Using Carat",0,6)
        data=data.loc[data["carat"]>=value]  
        fig,ax=plt.subplots(figsize=(10,5))
        sns.scatterplot(x="carat",y="price",hue="cut",data=data)              
        st.pyplot(fig)
    if graph=="Bar Graph":  
        fig,ax=plt.subplots(figsize=(5,3))
        sns.barplot(x="cut",y=data.cut.index,data=data)
        st.pyplot(fig)
    if graph=="Histogram":  
        fig,ax=plt.subplots(figsize=(5,3)) 
        sns.distplot(data.price,kde=True)
        st.pyplot(fig)
        