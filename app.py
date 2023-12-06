import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


vector=pickle.load(open('vector.pkl','rb'))
SV=pickle.load(open('model.pkl','rb'))

st.title("SMS Classifier")

input_sms = st.text_area("Enter your message")

def transform_message(message):
    message=message.lower()
    message=nltk.word_tokenize(message)

    y=[]
    for i in message:
        if i.isalnum():
            y.append(i)
    
    message=y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    message=y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))
    
    return " ".join(y)

if st.button('Predict'):
    if input_sms:
        transform_sms = transform_message(input_sms)

        vector_inp = vector.transform([transform_sms]).toarray()  

        result = SV.predict(vector_inp)[0]

        if result == 1:
            st.header('Wait a Minute, this is a SPAM!')
        else:
            st.header('Ohhh, this is a normal message.')
    else:
        st.warning("Please enter a message for prediction.")