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

    def extract_skills(self, text, skill_list):
        # This is a simple skill extraction. You might want to implement a more sophisticated method.
        skills = [skill for skill in skill_list if skill.lower() in text.lower()]
        return skills


# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    sample_text = "Je suis un ingénieur logiciel avec 5 ans d'expérience en Python et Machine Learning."
    processed_text = preprocessor.preprocess(sample_text)
    print(processed_text)

    skills = ["Python", "Machine Learning", "Java", "C++"]
    extracted_skills = preprocessor.extract_skills(sample_text, skills)
    print(f"Extracted skills: {extracted_skills}")