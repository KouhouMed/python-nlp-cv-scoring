import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self, language='french'):
        self.language = language
        self.stop_words = set(stopwords.words(self.language))
        self.lemmatizer = WordNetLemmatizer()

        def preprocess(self, text):
            # Convert to lowercase
            text = text.lower()

            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords and lemmatize
            cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

            # Join tokens back into string
            cleaned_text = ' '.join(cleaned_tokens)

            return cleaned_text