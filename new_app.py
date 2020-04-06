import pandas as pd
import numpy as np
import streamlit as st 
from PIL import Image

st.title("Image Enhancement")

from ISR.models import RDN

def enhanced(image1):
    model = RDN(weights='noise-cancel')
    sr_img_gan = model.predict(np.array(image1))
    return Image.fromarray(sr_img_gan)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Enhancing...")
    label = enhanced(uploaded_file)
    st.image(label, caption='Enhanced Image.', use_column_width=True)
    
