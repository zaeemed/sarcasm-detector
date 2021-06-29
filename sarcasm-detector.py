# libraries to be used
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
from keras.models import model_from_yaml

max_words = 1000
max_len = 150

# caching the load model function
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_RNN_model():
    # loading the RNN model from yaml file
    yaml_file = open('RNN_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    # learning saved weights of RNN
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("RNN_model.h5")
    # loading tokenizer model
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return loaded_model, tokenizer

# initializing the stepwords list
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# initializing lemmatizer and MinMaxScaler
lemm = WordNetLemmatizer()
scaler = MinMaxScaler()

# Adding Title to streamlit
st.markdown("<h1 style = 'color:Gold; Text-align: Center; font-size: 40px;'>Sarcasm Detector</h1>", unsafe_allow_html=True)

img = Image.open("./images/sarcasm.jpg")
# display image using streamlit
# width is used to set the width of an image
st.image(img, width = 700)

# loading the model
RNN_model,tokenizer = load_RNN_model()

# user input pre processing
def user_text_processing(user_text):
    user_text = user_text.split()
    user_text = [word.lower() for word in user_text if word not in stopwords]
    user_text = [lemm.lemmatize(word) for word in user_text]

    # converting user text to sequence
    user_seq = user_text
    user_seq = tokenizer.texts_to_sequences(user_seq)
    user_seq = sequence.pad_sequences(user_seq,maxlen=max_len)
    
    # min max scaling
    scaler.fit(user_seq)
    scaler.transform(user_seq)
    return user_seq

# user input prediction
def predict_sarcasm(user_seq):
#     prediction
    prob = RNN_model.predict(user_seq)
    probability = np.mean(prob, axis=0)
    # probability
    if probability > 0.5:
        st.success(f"Sentence '{user_text}' is of 'Sarcastic' nature")
        img = Image.open("./images/sarcastic.png")
        st.image(img, width = 600)
        
    elif probability < 0.5:
        st.success(f"Sentence '{user_text}' is of 'Non Sarcastic / Acclaim' nature")
        img = Image.open("./images/acclaim.png")
        st.image(img, width = 600)
        
    elif probability == 0.5:
        st.success(f"Sentence '{user_text}' is of 'Neutral' nature")
        img = Image.open("./images/neutral.gif")
        st.image(img, width = 700)
    
    
# taking user input
form = st.form(key='my_form')
user_text = form.text_input(label='Enter Your Text')
submit = form.form_submit_button(label= 'Predict Sarcasm' )
st.write('Press Predict Sarcasm to Predict your text nature')
if submit:
    user_seq = user_text_processing(user_text)
    prediction = predict_sarcasm(user_seq)
