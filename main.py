import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}



model = load_model('simple_rnn_model_imdb.h5')

# Helper Functions
# Function to decode the reviews
def decode_review(encoded_review):
    ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess our inputs or reviews
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    pre_processed_input = preprocess_text(review)
    prediction = model.predict(pre_processed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

st.title('Sentiment Analysis for IMDB Review Dataset')

user_input = st.text_area('Movie Review')
if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    preidition = model.predict(preprocessed_input)
    sentiment = 'Positive' if preidition[0][0] > 0.5 else 'Negative'

    # Display the results
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {preidition[0][0]}')

else:
    st.write('Please enter a movie review')

