import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
from sklearn.preprocessing import OneHotEncoder

##importing data from file
filepath = 'C:/Users/rajij/Streamlit_Home Page/Industrial copper/Copper_Set_Data.csv'
df = pd.read_csv(filepath)

item_type_d = df['item type'].unique()
application_d = df['application'].unique()
country_d = df['country'].unique()
product_ref_d = df['product_ref'].unique()
material_ref_d = df['material_ref'].unique()
status_d = df['status'].unique()

##Cleaning and transforming data

df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['material_ref'].fillna('NULL', inplace=True)
df['country'].mode()
df['country'].fillna(78.0, inplace=True)
df['application'].mode()
df['application'].fillna(10.0, inplace=True)
df = df.dropna()
df['material_ref'] = df['material_ref'].replace(to_replace=r'^000000000000000000000000000000000.*', value=np.nan, regex=True)
df['material_ref'] = df['material_ref'].replace(to_replace=r'^0000000000000000000000000000000.*', value=np.nan, regex=True)
df['material_ref'].fillna('NULL', inplace=True)
df['material_ref'] = df['material_ref'].apply(lambda x: 'others' if x.startswith('000') else x)
value_counts = df['material_ref'].value_counts()
freq = (value_counts[value_counts<100]).index
unique_once_list = list(freq)
df['material_ref'] = df['material_ref'].replace(unique_once_list, 'other')
df = df[df['status'].isin(['Won', 'Lost'])]


OE = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


df['status'] = OE.fit_transform(df[['status']])


OE1 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

df['item type'] = OE1.fit_transform(df[['item type']])

OE2 = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df['material_ref'] = OE2.fit_transform(df[['material_ref']])

OE3 = OrdinalEncoder()
df['product_ref'] = OE3.fit_transform(df[['product_ref']])


df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')

df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date

df['Delivery_time'] = df['delivery date'] - df['item_date']
df['Delivery_time'] = df['Delivery_time'].astype(str)
##lot of incorrect values..ignor
df['Delivery_time'] = df['Delivery_time'].fillna(0).astype(str)

df['Delivery_time'] = (df['Delivery_time'].str.extract('(\d+)'))
df['Delivery_time'] = pd.to_numeric(df['Delivery_time'], errors='coerce')

df = df.dropna()


features = df[['width', 'selling_price', 'thickness', 'application', 'quantity tons']]

# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1)

# Fit the model
iso_forest.fit(features)

# Predict anomalies (-1 for anomalies, 1 for normal points)
df['anomaly'] = iso_forest.predict(features)
cleaned_data_ = df[df['anomaly'] != -1]
##removing values less than or equal to zero
a = cleaned_data_['selling_price'] <= 0
print(a.sum())
#cleaned_data.loc[a, 'selling_price'] = np.nan

a = cleaned_data_['quantity tons'] <= 0
print(a.sum())

#cleaned_data_if.loc[a, 'quantity tons'] = np.nan

a = cleaned_data_['thickness'] <= 0
print(a.sum())

a = cleaned_data_['width'] <= 0
print(a.sum())

a = cleaned_data_['Delivery_time'] <= 0
print(a.sum())

cleaned_data_ = cleaned_data_[cleaned_data_['selling_price'] > 0]
cleaned_data_ = cleaned_data_[cleaned_data_['quantity tons'] > 0]
cleaned_data_ = cleaned_data_[cleaned_data_['thickness'] > 0]
cleaned_data_ = cleaned_data_[cleaned_data_['width'] > 0]
cleaned_data_ = cleaned_data_[cleaned_data_['Delivery_time'] > 0]

##data transformation using log function
cleaned_data_['quantity tons_log'] = np.log(cleaned_data_['quantity tons'])
cleaned_data_['selling_price_log'] = np.log(cleaned_data_['selling_price'])
cleaned_data_['thickness_log'] = np.log(cleaned_data_['thickness'])
cleaned_data_['width_log'] = np.log(cleaned_data_['width'])
cleaned_data_ = cleaned_data_.dropna(subset=['selling_price_log'])
cleaned_data_['Delivery_time_log'] = np.log(cleaned_data_['Delivery_time'])





## Regression modelling

#split data into X, y
x=cleaned_data_[['application','thickness_log','width_log','country','product_ref', 'quantity tons_log','customer','status','item type', 'material_ref', 'Delivery_time_log']]
y=cleaned_data_['selling_price_log']

scaler = StandardScaler()
scaler.fit(x)
scaler.transform(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=123)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

##Classifier Modelling 
a=cleaned_data_[['selling_price_log','item type','application','thickness_log','width_log','country','product_ref', 'material_ref', 'quantity tons_log','customer', 'Delivery_time_log' ]]
b=cleaned_data_['status' ]

scaler_c = StandardScaler()
scaler_c.fit(a)
scaler_c.transform(a)

from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a,b, test_size = 0.2, random_state=125, stratify=b)

dc = DecisionTreeClassifier(criterion= 'entropy',max_depth= 7, min_samples_split= 14)
dc.fit(a_train, b_train)


##Saving files using pickle

with open('randomeforestreg.pkl', 'wb') as file:
    pickle.dump(rf, file)

with open('encoder.pkl', 'wb') as file2:
    pickle.dump(OE, file2)

with open('encoder1.pkl', 'wb') as file2:
    pickle.dump(OE1, file2)

with open('encoder2.pkl', 'wb') as file2:
    pickle.dump(OE2, file2)

with open('encoder3.pkl', 'wb') as file2:
    pickle.dump(OE3, file2)

with open('scaler.pkl', 'wb') as file3:
    pickle.dump(scaler, file3)

with open('decisiontreeclass.pkl', 'wb') as file4:
    pickle.dump(dc, file4)   

with open('scaler1.pkl', 'wb') as file5:
    pickle.dump(scaler_c, file5)

st.title("IndustrialCopperModeling")


