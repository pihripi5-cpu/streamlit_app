import streamlit as st
st.title("Checking the person eleigible for vate or not ")

age=st.number_input("Enter your age...")

if st.button("Submit"):
    if age>=18:
        st.title("Eligible for Vote...")
    else:
        st.title("Not eligible for Vote")
