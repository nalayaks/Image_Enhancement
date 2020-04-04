import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber Pickups in New York City")
st.markdown(
"""
This is a demo of a Streamlit app that shows the Uber pickups
geographical distribution in New York City. Use the slider
to pick a specific hour and look at how the charts change.

[See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
""")
