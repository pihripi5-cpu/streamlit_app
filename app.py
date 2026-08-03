import streamlit as st
st.title("My Page On  Streamlit application")

name=st.text_input("Enter your name")

if st.button("Submit"):
  st.write(f"Hello,{name}")
