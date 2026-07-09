import streamlit as st


def square(x):
    return x * x

num = st.number_input("Enter a number", value=0)

if st.button("Calculate Square"):
    result = square(num)
    st.write(f"The square of {num} is {result}.")   