import streamlit as st
import pickle
import zipfile
import nltk
from PIL import Image
import io
import pytesseract
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import requests


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
vector = pickle.load(open('vector.pkl', 'rb'))

# Function to load compressed pickle file
def load_compressed_pickle(zip_filename, pickle_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        with zip_ref.open(pickle_filename) as pickle_file:
            loaded_data = pickle.load(pickle_file)
    return loaded_data

zip_filename = 'model.zip'
pickle_filename = 'model.pkl'
SV = load_compressed_pickle(zip_filename, pickle_filename)

# Function to transform the message
def transform_message(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = [i for i in message if i.isalnum()]
    message = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    y = [i for i in message if i not in stop_words and i not in string.punctuation]
    message = y[:]
    y.clear()

    y = [ps.stem(i) for i in message]
    return " ".join(y)

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select a page:",["Text Classify", "Text Input", "Image Input", "Feedback"])

# Page: Text Classify
if option == "Text Classify":
    st.title("Text Classify🔍")
    st.write("Welcome to the Text Classify!")
    st.write("This application classifies messages into spam or normal.")
    st.write("### How it Works")
    st.write("1. **Text Classification**: You can enter a text message, and the app will analyze it to determine whether it is spam or a normal message based on its content.")
    st.write("2. **Image Input**: Upload an image containing text, and the app will extract the text from the image and classify it accordingly.")
    st.write("3. **Feedback**: Provide feedback to help improve the accuracy and functionality of the application.")
    st.write("### Features")
    st.write("1. High accuracy classification using advanced machine learning models.")
    st.write("2. User-friendly interface for both text and image input.")
    
# Page: Text Input
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
                    st.header('Be Careful, Its a SPAM Message!')
                else:
                    st.header('Okay, this is a normal message.')
        else:
            st.warning("Please enter a message for prediction.")

# Page: Image Input
elif option == "Image Input":
    st.title("Image Input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        with st.spinner('Processing...'):
            # Extract text from the image using pytesseract
            text = pytesseract.image_to_string(image)
            st.text_area("Extracted Text", text)
            
            if text:
                transform_sms = transform_message(text)
                vector_inp = vector.transform([transform_sms]).toarray()
                result = SV.predict(vector_inp)[0]
                if result == 1:
                    st.header('Be Careful, Its a SPAM Message!')
                else:
                    st.header('Okay, this is a normal message.')
            else:
                st.warning("No text detected in the image.")

# Page: Feedback
elif option == "Feedback":
    st.title("Feedback")
    feedback = st.text_area("Your feedback")
    
    if st.button('Submit Feedback'):
        if feedback:
            with st.spinner('Submitting...'):
                formspree_url = 'https://formspree.io/f/xwpevovv'
                data = {'message': feedback}
                response = requests.post(formspree_url, data=data)
                
                if response.status_code == 200:
                    st.success("Thank you for your feedback!")
                else:
                    st.error("There was an error submitting your feedback. Please try again later.")
        else:
            st.warning("Please provide your feedback before submitting.")
