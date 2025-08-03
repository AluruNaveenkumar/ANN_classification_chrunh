#Import necessary Lib's

import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
import pickle

# Load the files
model=tf.keras.models.load_model('model.h5')

#Load the Other files
with open('lable_encoder_gender.pkl','rb') as file:
    lable_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler_value=pickle.load(file)

#Streamlite implementation
st.title("Custmore Churn Prediction")

#user inputs
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',lable_encoder_gender.classes_)
age = st.slider('Age',18,100)
balance = st.number_input('Balance')
credit_score = st.number_input('CreditScore')
estimated_salary = st.number_input('EstimatedSalary')
tenure = st.number_input('Tenure',0,10)
num_of_products = st.number_input('NumOfProducts',1,4)
has_cr_card = st.selectbox('HasCrCard',[0,1])
is_active_member = st.selectbox('IsActiveMember',[0,1])

input_data=pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [lable_encoder_gender.transform([gender])],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]

})

geo=onehot_encoder_geo.transform([[geography]])
geos=onehot_encoder_geo.get_feature_names_out()

df1=pd.DataFrame(data=geo,columns=geos)

data=pd.concat([input_data,df1],axis=1)

scaled=scaler_value.transform(data)

#model prediction
prediction=model.predict(scaled)

if prediction[0][0] > 0.5:
    st.write("Customer is like to continue servies")
else:
    st.write("Customer gone to Leave the services")