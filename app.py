import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Load the model
model_path = r'sentiment_analysis (3).pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess_review(review):
    review = remove_html_tags(review)
    review = review.lower()
    review = remove_stopwords(review)
    return review

# Streamlit app
st.title('Sentiment Analysis Model')

review = st.text_input('Enter your review:')
submit = st.button('Predict')

if submit:
    processed_review = preprocess_review(review)
    prediction = model.predict([processed_review])

    if prediction[0] == 'positive':
        st.success('Positive Review')
    else:
        st.warning('Negative Review')
