import pandas as pd
import numpy as np
import streamlit as st 
from PIL import Image
import keras
from ISR.models import RRDN
rrdn = RRDN(weights='gans')

st.title("Image Enhancement")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Enhancing...")
    sr_img_gan = rrdn.predict(uploaded_file)
    sr_img_gan
