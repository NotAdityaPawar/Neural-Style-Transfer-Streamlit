import tensorflow as tf
import tensorflow_hub as hub
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

from PIL import Image

model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

def load_image(img):

    #img = tf.io.read_file(img)
    img = tf.image.decode_image(img,channels = 3)
    img = tf.image.convert_image_dtype(image=img,dtype=tf.float32)
    img = img[tf.newaxis,:]
    return img

st.title("Neural Style Transfer")

col1, col2 = st.columns(2)

with col1:
    style = st.file_uploader('Upload the style file')
    print(style)
    st.write(style)
    st.image(style)

    data = style.getvalue()
    style = load_image(data)

with col2:
    person  = st.file_uploader("Upload the image")
    st.write(person)
    # print(person)
    st.image(person)

    
    data = person.getvalue()
    # st.write(data)
    person = load_image(data)

    if st.button("Style"):
        styled_img = model(tf.constant(person),tf.constant(style))[0]

        #st.write(type(styled_img))
        st.image(np.squeeze(styled_img))
        # st.image(styled_img)

        st.stop()
        


    