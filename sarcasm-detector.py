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
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
stopwords = nltk.corpus.stopwords.words('english')
lemm = WordNetLemmatizer()
scaler = MinMaxScaler()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_RNN_model():
    yaml_file = open('RNN_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("RNN_model.h5")
    print("Loaded model from disk")
    nltk.download('stopwords')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return loaded_model, tokenizer
# Title
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
    user_seq = np.array(user_text)
    user_seq = tokenizer.texts_to_sequences(user_seq)
    user_seq = sequence.pad_sequences(user_seq,maxlen=max_len)
    
    # min max scaling
    scaler.fit(user_seq)
    scaler.transform(user_seq)
    return user_seq

def predict_sarcasm(user_seq):
#     prediction
    prob = RNN_model.predict(user_seq)
    probability = np.mean(prob, axis=0)
    probability
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
