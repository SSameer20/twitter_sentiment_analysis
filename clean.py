import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Check if stopwords data is available, and download it if not
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

# Load the data
train_data = pd.read_csv('./archive/twitter_training.csv')
validate_data = pd.read_csv('./archive/twitter_validation.csv')

# Preprocessing function
def preprocess_text(text):
    # Check if the text is NaN or not a string
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove mentions (usernames)
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    
    # Remove non-alphabetical characters (like numbers and punctuation)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Load stopwords only once
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text