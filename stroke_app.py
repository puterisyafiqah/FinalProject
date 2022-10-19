# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:24:55 2022

@author: Acer
"""

# WebApp for Stroke Detection
# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.title('Stroke Prediction')
st.info('All data is available at: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset')
st.subheader("Data Information")
st.dataframe(df)
st.subheader("Statistical Information")
st.write(df.describe())
st.write("samplechart")
fig, ax = plt.subplots()
ax.hist(df['bmi'])

st.pyplot(fig)
stroke = {0:'Not Stroke',1:'Stroke'}
st.bar_chart(data='smoking status', *, x=smoking status, y=none, width=6, height=8, use_container_width=True)

gender = st.sidebar.radio('Gender', ('Male', 'Female'))
age = st.sidebar.number_input('Age (years)', value = 30, min_value = 0, max_value = 100)
hypertension = st.sidebar.radio('Hypertension', (1, 0))
heart_disease = st.sidebar.radio('Heart Disease', (1, 0))
ever_married = st.sidebar.radio('Ever Married', ('Yes', 'No'))
work_type = st.sidebar.multiselect('Work Type', options = ['children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'])
residence_type = st.sidebar.radio('Residence Type', ('Rural', 'Urban'))
avg_glucose_level = st.sidebar.slider('Average Glucose Level', min_value = 50.0, max_value = 300.0)
bmi = st.sidebar.slider('Body Mass Index', min_value = 5.0, max_value = 70.0)
smoking_status = st.sidebar.multiselect('Smoking Status', options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

gender = {"Male": 0, "Female": 1}
ever_married_label={'No': 0, 'Yes': 1} 
work_type= {'Private': 0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4}
residence_type={'Urban': 0, 'Rural': 1}
smoking_status={'formerly smoked': 0, 'never smoked': 1, 'smokes':2, 'Unknown':3}

df['bmi'].fillna(df['bmi'].median(),inplace=True)
df=df.drop('id',axis=1)
df.drop(df[df['gender'] == 'Other'].index, inplace = True)
categorical_data=df.select_dtypes(include=['object']).columns
le=LabelEncoder()
df[categorical_data]=df[categorical_data].apply(le.fit_transform)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if st.button('Predict') == True:
    rf = RandomForestClassifier(random_state=25)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    if y_pred == 1:
        st.success('Stroke')
    else:
        st.snow() # effect of snow
        st.warning('Not Stroke')
