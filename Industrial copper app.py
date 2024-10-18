import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(layout='wide')

with st.sidebar:
    select= option_menu("Main Page",["HomePage","Modelling"])

if select =="HomePage":
 st.title("IndustrialCopperModeling")    

elif select=='Modelling':
 tab1,tab2= st.tabs(["Predict Selling Price","Predict Status"])

##For Regression Modelling: 
 with tab1: 

# User input for features
   st.write("Input the details below:")

   filepath = 'C:/Users/rajij/Streamlit_Home Page/Industrial copper/Copper_Set_Data.csv'
   df = pd.read_csv(filepath)

   item_type_d = df['item type'].unique()
   application_d = df['application'].unique()
   country_d = df['country'].unique()
   product_ref_d = df['product_ref'].unique()
   material_ref_d = df['material_ref'].unique()
   status_d = df['status'].unique()

   col1, col2 = st.columns([1,1])
   with col1:
      Item_type_ = st.selectbox('Item_type', item_type_d, index=None, key=1)
      application_ = st.selectbox('application', application_d, index=None, key=2)

      country_ = st.selectbox('country', country_d, index=None, key=3)
      product_ref_ = st.selectbox('product_ref', product_ref_d, index=None, key=4)
      status_ = st.selectbox('status', status_d, index=None, key=5)
   

   with col2: 
      quantity_tons_ = st.number_input("quantity tons", min_value=0.0, max_value=10000.00)

      thickness_ = st.number_input("thickness", min_value=0.0, max_value=100.0)
      
      width_ = st.number_input("width", min_value=0.0, max_value=10000.0)

      Delivery_time_ = st.number_input("Delivery_time", min_value=0.0, max_value=20000.00)
      
      customer_ = st.number_input("customer")

      material_ref_ = st.text_input("material_ref")
 

   if st.button("Predict"):
      if not quantity_tons_ or not thickness_ or not width_ or not customer_ or not material_ref_  or not Delivery_time_ or Item_type_ is None or application_ is None or country_ is None or product_ref_ is None or status_ is None:
         st.error('Please fill in all the above fields!')
      else:


         with open('randomeforestreg.pkl', 'rb') as file:
            rf = pickle.load(file)

         with open('encoder.pkl', 'rb') as file2:
            loaded_Encoder = pickle.load(file2)

         with open('encoder1.pkl', 'rb') as file2:
            loaded_Encoder1 = pickle.load(file2)

         with open('encoder2.pkl', 'rb') as file2:
            loaded_Encoder2 = pickle.load(file2)

         with open('encoder3.pkl', 'rb') as file2:
            loaded_Encoder3 = pickle.load(file2)


         with open('scaler.pkl', 'rb') as file3:
            loaded_scaler = pickle.load(file3)


         input_data = {
               'item type': [Item_type_],
               'application': [application_],
               'country': [country_],
               'product_ref': [product_ref_],
               'status': [status_],
               'quantity tons_log': [quantity_tons_],
               'thickness_log': [thickness_],
               'width_log': [width_],
               'customer': [customer_],
               'material_ref': [material_ref_],
               'Delivery_time_log': [Delivery_time_]       
            }

         input_data = pd.DataFrame(input_data)

#st.dataframe(input_data)

         input_data['status'] = loaded_Encoder.transform(input_data[['status']])
         input_data['item type'] = loaded_Encoder1.transform(input_data[['item type']])
         input_data['material_ref'] = loaded_Encoder2.transform(input_data[['material_ref']])
         input_data['product_ref'] = loaded_Encoder3.transform(input_data[['product_ref']])
         input_data['width_log'] = np.log(input_data['width_log'])
         input_data['quantity tons_log'] = np.log(input_data['quantity tons_log'])
         input_data['thickness_log'] = np.log(input_data['thickness_log'])
         input_data['Delivery_time_log'] = np.log(input_data['Delivery_time_log'])

         input_data=input_data[['application','thickness_log','width_log','country','product_ref', 'quantity tons_log','customer','status','item type', 'material_ref', 'Delivery_time_log']]

#st.dataframe(input_data)
         input_data_s = loaded_scaler.transform(input_data)     
         new_pred = rf.predict(input_data_s)[0]
         st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

##for classification modelling
 with tab2: 

# User input for features
   st.write("Input the details below:")

   filepath = 'C:/Users/rajij/Streamlit_Home Page/Industrial copper/Copper_Set_Data.csv'
   df = pd.read_csv(filepath)

   item_type_d = df['item type'].unique()
   application_d = df['application'].unique()
   country_d = df['country'].unique()
   product_ref_d = df['product_ref'].unique()
   material_ref_d = df['material_ref'].unique()
   status_d = df['status'].unique()

   col1, col2 = st.columns([1,1])
   with col1:
      Item_type_ = st.selectbox('Item_type', item_type_d, index=None, key=9)
      application_ = st.selectbox('application', application_d, index=None, key=6)
      
      country_ = st.selectbox('country', country_d, index=None, key=7)
      product_ref_ = st.selectbox('product_ref', product_ref_d, index=None, key=8)
      selling_price_ = st.number_input("selling_price", min_value=0.0, max_value=10000000.00)
   

   with col2: 
      quantity_tons_ = st.number_input("quantity tons", min_value=0.0, max_value=10000000.00)

      thickness_ = st.number_input("thickness", min_value=0.0, max_value=10000000.00)
      
      width_ = st.number_input("width", min_value=0.0, max_value=10000000.00)

      Delivery_time_ = st.number_input("Delivery_time", min_value=0.0, max_value=10000.00)
      
      customer_ = st.number_input("customer", min_value=0.0, max_value=100000000.00)

      material_ref_ = st.text_input("material_ref", key=22)
 

   if st.button("Predict Status"):

      if not quantity_tons_ or not thickness_ or not width_ or not customer_ or not material_ref_  or not selling_price_ or Item_type_ is None or application_ is None or country_ is None or product_ref_ is None:
         st.error('Please fill in all the above fields!')
      else:


         with open('encoder1.pkl', 'rb') as file2:
            loaded_Encoder1 = pickle.load(file2)

         with open('encoder2.pkl', 'rb') as file2:
            loaded_Encoder2 = pickle.load(file2)

         with open('encoder3.pkl', 'rb') as file2:
            loaded_Encoder3 = pickle.load(file2)

         with open('decisiontreeclass.pkl', 'rb') as file4:
            loaded_dc = pickle.load(file4)

         with open('scaler1.pkl', 'rb') as file5:
            loaded_scaler1 = pickle.load(file5)

         input_data = {
               'item type': [Item_type_],
               'application': [application_],
               'country': [country_],
               'product_ref': [product_ref_],
               'selling_price_log': [selling_price_],
               'quantity tons_log': [quantity_tons_],
               'thickness_log': [thickness_],
               'width_log': [width_],
               'customer': [customer_],
               'material_ref': [material_ref_],
               'Delivery_time_log': [Delivery_time_]        
            }

         input_data = pd.DataFrame(input_data)

#st.dataframe(input_data)
         
         input_data['item type'] = loaded_Encoder1.transform(input_data[['item type']])
         input_data['material_ref'] = loaded_Encoder2.transform(input_data[['material_ref']])
         input_data['product_ref'] = loaded_Encoder3.transform(input_data[['product_ref']])
         input_data['width_log'] = np.log(input_data['width_log'])
         input_data['quantity tons_log'] = np.log(input_data['quantity tons_log'])
         input_data['thickness_log'] = np.log(input_data['thickness_log'])
         input_data['selling_price_log'] = np.log(input_data['selling_price_log']) 
         input_data['Delivery_time_log'] = np.log(input_data['Delivery_time_log'])
         input_data=input_data[['selling_price_log','item type','application','thickness_log','width_log','country','product_ref', 'material_ref', 'quantity tons_log','customer', 'Delivery_time_log' ]]

#st.dataframe(input_data)
         input_data_s = loaded_scaler1.transform(input_data)     
         new_pred_ = loaded_dc.predict(input_data_s)
#st.write(new_pred_)
         if new_pred_.all()==1.0:
            st.write('## :green[The Status is Won] ')
         else:
            st.write('## :red[The status is Lost] ')
           

# # EDA Visualization
# st.subheader("Data Visualization")