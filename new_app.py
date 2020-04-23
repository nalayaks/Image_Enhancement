import pandas as pd
import numpy as np
import streamlit as st 
from PIL import Image
import ISR
import keras

st.title("Image Enhancement")

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from ISR.models import RDN
model = RDN(weights='noise-cancel')

#@st.cache(suppress_st_warning=True)
def enhanced(image1):
    ###
    ###
    image = load_img(image1)
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    sr_img_gan = model.predict(image)
    return Image.fromarray(sr_img_gan)

#def im2double(im):
#    if im.dtype == 'uint8':
#        out = im.astype('float') / 255
#    elif im.dtype == 'uint16':
#        out = im.astype('float') / 65535
#    elif im.dtype == 'float':
#        out = im
#    else:
#        assert False
#    out = np.clip(out, 0, 1)
 #   return out
    
#import cv2
 
# Opens the Video file
#cap= cv2.VideoCapture('/Users/sysadmin/Downloads/Video Enhancement unit/Fail/abids/Bharat Interior_shivaji bridge-2020-02-25_05h40min00s000ms.mp4')
#i=0
#while(cap.isOpened()):
#    ret, frame = cap.read()
#    if ret == False:
#        break
#    cv2.imwrite('kang'+str(i)+'.jpg',frame)
#    i+=1
 
#cap.release()
#cv2.destroyAllWindows()
#cv2.waitKey()     


uploaded_file = st.file_uploader("Choose an image....", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Enhancing...")
    label = enhanced(uploaded_file)
    st.image(label, caption='Enhanced Image.', use_column_width=True)
    
