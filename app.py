import streamlit as st
st.title("My stream lit app")
 names=st.text_input("Enter your name")

if st.button("Submit"):
  st.write(f"Hello,{name}")
