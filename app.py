import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

st.header('HealthyClassifier / Image Classification Model')
model = load_model('C:\\Python34\\Image_Classification\\Image_classify.keras')
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
calories = {
    'apple': 52, 'banana': 96, 'beetroot': 43, 'bell pepper': 31, 'cabbage': 25, 'capsicum': 40, 'carrot': 41, 'cauliflower': 25, 'chilli pepper': 40, 'corn': 86, 'cucumber': 16, 'eggplant': 25, 'garlic': 149, 'ginger': 80, 'grapes': 69, 'jalepeno': 29, 'kiwi': 61, 'lemon': 29, 'lettuce': 15, 'mango': 60, 'onion': 40, 'orange': 47, 'paprika': 282, 'pear': 57, 'peas': 81, 'pineapple': 50, 'pomegranate': 83, 'potato': 77, 'raddish': 16, 'soy beans': 446, 'spinach': 23, 'sweetcorn': 86, 'sweetpotato': 86, 'tomato': 18, 'turnip': 28, 'watermelon': 30
}

img_height = 180
img_width = 180
image = st.text_input('Enter Image name ', 'Apple.jpg')
image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, 0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
predicted_item = data_cat[np.argmax(score)]
st.write(f'The Vegetable/Fruit in image is {predicted_item}')
st.write(f'With accuracy of {np.max(score) * 100:.2f}%')

calorie_info = calories.get(predicted_item, 'Calorie information not available')
if isinstance(calorie_info, int):
    st.write(f'Calories per 100g: {calorie_info} kcal')
else:
    st.write(calorie_info)
