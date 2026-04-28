import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

# Load the trained model
model = pk.load(open('model.pkl','rb'))

st.header('Car Price Prediction ML Model')

# Lock the select boxes to ONLY the categories the model recognizes
name = st.selectbox('Select Car Brand', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'])
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
seller_type = st.selectbox('Seller type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission type', ['Manual', 'Automatic'])
owner = st.selectbox('Owner type', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.slider('Car Mileage', 10, 40)
engine = st.slider('Engine CC', 700, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 5, 10)

if st.button("Predict"):
    # 1. Map text to numbers using Python dictionaries (Much safer than Pandas .replace)
    name_map = {'Maruti': 1, 'Skoda': 2, 'Honda': 3, 'Hyundai': 4, 'Toyota': 5, 'Ford': 6, 'Renault': 7,
                'Mahindra': 8, 'Tata': 9, 'Chevrolet': 10, 'Datsun': 11, 'Jeep': 12, 'Mercedes-Benz': 13,
                'Mitsubishi': 14, 'Audi': 15, 'Volkswagen': 16, 'BMW': 17, 'Nissan': 18, 'Lexus': 19,
                'Jaguar': 20, 'Land': 21, 'MG': 22, 'Volvo': 23, 'Daewoo': 24, 'Kia': 25, 'Fiat': 26, 
                'Force': 27, 'Ambassador': 28, 'Ashok': 29, 'Isuzu': 30, 'Opel': 31}
    
    fuel_map = {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4}
    
    seller_map = {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3}
    
    trans_map = {'Manual': 1, 'Automatic': 2}
    
    owner_map = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}

    # 2. Convert the user inputs using the maps
    mapped_name = name_map[name]
    mapped_fuel = fuel_map[fuel]
    mapped_seller = seller_map[seller_type]
    mapped_trans = trans_map[transmission]
    mapped_owner = owner_map[owner]

    # 3. Create the DataFrame using the converted numbers directly
    input_data_model = pd.DataFrame(
        [[mapped_name, year, km_driven, mapped_fuel, mapped_seller, mapped_trans, mapped_owner, mileage, engine, max_power, seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']
    )

    # 4. Convert all columns to float (Since they are all numbers now, this will never fail!)
    input_data_model = input_data_model.astype(float)

    # 5. Run the prediction
    car_price = model.predict(input_data_model)

    # 6. Display the result
    st.success('Estimated Car Price: ₹ ' + str(round(car_price[0], 2)))
