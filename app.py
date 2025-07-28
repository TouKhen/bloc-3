import streamlit as st
import pandas as pd
import numpy as np

st.write('Hello World!')
data = pd.read_csv('./data/telco-customer-churn.csv')

st.write('Telco Customer Churn')
st.write(data)