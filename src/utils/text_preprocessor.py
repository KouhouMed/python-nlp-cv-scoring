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