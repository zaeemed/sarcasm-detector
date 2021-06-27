from PIL import Image
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
import nltk

max_words = 1000
max_len = 150
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

stopwords = nltk.corpus.stopwords.words('english')
lemm = WordNetLemmatizer()
scaler = MinMaxScaler()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    RF_model = joblib.load('./RF_model.joblib')
    return RF_model

# Title
st.markdown("<h1 style = 'color:Gold; Text-align: Center; font-size: 40px;'>Sarcasm Detector</h1>", unsafe_allow_html=True)

img = Image.open("./images/sarcasm.jpg")
  
# display image using streamlit
# width is used to set the width of an image
st.image(img, width = 700)

# loading the model
RF_model = load_model()

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
    prob = RF_model.predict_proba(user_seq)
    probability = np.mean(prob, axis=0)
    if probability[0] < probability[1]:
        st.success(f"Sentence '{user_text}' is of 'Sarcastic' nature")
        img = Image.open("./images/sarcastic.png")
        st.image(img, width = 700)

    elif probability[0] > probability[1]:
        st.success(f"Sentence '{user_text}' is of 'Non Sarcastic / Acclaim' nature")
        img = Image.open("./images/acclaim.png")
        st.image(img, width = 700)
        
    elif probability[0] == probability[1]:
        st.success(f"Sentence '{user_text}' is of 'Neutral' nature")
        img = Image.open("./images/neutral.gif")
        st.image(img, width = 700)

    
# taking user input
form = st.form(key='my_form')
user_text = form.text_input(label='Enter Your Text')
submit = form.form_submit_button(label= 'Predict' )
st.write('Press Predict to Predict your text nature')

if submit:
    user_seq = user_text_processing(user_text)
    prediction = predict_sarcasm(user_seq)