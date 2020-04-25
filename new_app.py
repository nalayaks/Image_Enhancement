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

# April 22, 2019
# Tran Le Anh - MSc Student in Computer Vision
# Dept. of Electronics Engineering, Myongji University, South Korea
# tranleanh.nt@gmail.com
# https://sites.google.com/view/leanhtran

import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt

def blur(img, kernel_size = 3):
	dummy = np.copy(img)
	h = np.eye(kernel_size) / kernel_size
	dummy = convolve2d(dummy, h, mode = 'valid')
	return dummy

def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
	# Load image and convert it to gray scale
	file_name = os.path.join('lena.jpg') 
	img = rgb2gray(plt.imread(file_name))

	# Blur the image
	blurred_img = blur(img, kernel_size = 15)

	# Add Gaussian noise
	noisy_img = add_gaussian_noise(blurred_img, sigma = 20)

	# Apply Wiener Filter
	kernel = gaussian_kernel(3)
	filtered_img = wiener_filter(noisy_img, kernel, K = 10)


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
    
