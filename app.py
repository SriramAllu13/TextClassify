import streamlit as st
import pickle
import zipfile
import nltk
from PIL import Image
import pytesseract
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt.zip')
except LookupError:
    nltk.download('punkt')

# Load the vector and model
ps = PorterStemmer()
vector = pickle.load(open('vector.pkl', 'rb'))

def load_compressed_pickle(zip_filename, pickle_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        with zip_ref.open(pickle_filename) as pickle_file:
            loaded_data = pickle.load(pickle_file)
    return loaded_data

zip_filename = 'model.zip'
pickle_filename = 'model.pkl'
SV = load_compressed_pickle(zip_filename, pickle_filename)

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option:", ["Text Classify", "Text Input", "Image Input", "Feedback"])

# Function to transform message
def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = [i for i in message if i.isalnum()]
    message = y[:]
    y.clear()

    y = [i for i in message if i not in stopwords.words('english') and i not in string.punctuation]
    message = y[:]
    y.clear()

    y = [ps.stem(i) for i in message]
    return " ".join(y)

# Text Classify
if option == "Text Classify":
    st.title("SMS Classifier")
    st.write("This is a spam detection system for SMS messages.")
    st.write("The system classifies messages into spam or normal based on the content.")
    
elif option == "Text Input":
    st.title("Text Input")
    input_sms = st.text_area("Enter your message")

    if st.button('Predict'):
        if input_sms:
            with st.spinner('Processing...'):
                transform_sms = transform_message(input_sms)
                vector_inp = vector.transform([transform_sms]).toarray()
                result = SV.predict(vector_inp)[0]

                if result == 1:
                    st.header('Wait a Minute, this is a SPAM!')
                else:
                    st.header('Ohhh, this is a normal message.')
        else:
            st.warning("Please enter a message for prediction.")
    
elif option == "Image Input":
    st.title("Image Input")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Processing...")
        
        with st.spinner('Extracting text from image...'):
            image = Image.open(uploaded_image)
            text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text", text, height=200)
            
            if st.button('Predict from Extracted Text'):
                if text:
                    transform_sms = transform_message(text)
                    vector_inp = vector.transform([transform_sms]).toarray()
                    result = SV.predict(vector_inp)[0]

                    if result == 1:
                        st.header('Wait a Minute, this is a SPAM!')
                    else:
                        st.header('Ohhh, this is a normal message.')
                else:
                    st.warning("No text extracted from image.")
    
elif option == "Feedback":
    st.title("Feedback")
    feedback = st.text_area("Please provide your feedback here:")

    if st.button('Submit Feedback'):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Feedback cannot be empty.")

