import json
import random
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    """
    Tokenize, lemmatize, and remove stopwords using spaCy.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

class Chatbot:
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.intents = self.data['intents']
        self.vectorizer = TfidfVectorizer()
        self.prepare_data()

    def prepare_data(self):
        """
        Prepare questions for TF-IDF vectorization.
        """
        self.all_patterns = []
        self.pattern_to_tag = []
        
        for intent in self.intents:
            for pattern in intent['patterns']:
                preprocessed_pattern = preprocess(pattern)
                if preprocessed_pattern: # Ensure it's not empty after preprocessing
                    self.all_patterns.append(preprocessed_pattern)
                    self.pattern_to_tag.append(intent['tag'])
        
        # Fit and transform the patterns
        self.tfidf_matrix = self.vectorizer.fit_transform(self.all_patterns)

    def get_sentiment(self, text):
        """
        Detect basic sentiment of the user input.
        """
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0.1:
            return "Positive"
        elif analysis.sentiment.polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"

    def get_response(self, user_input):
        """
        Match user input with the best dataset response.
        """
        preprocessed_input = preprocess(user_input)
        if not preprocessed_input:
            return "I'm sorry, I didn't quite catch that. Could you please rephrase?", "Neutral"

        # Vectorize user input
        user_vector = self.vectorizer.transform([preprocessed_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        
        # Get the best match
        max_similarity = np.max(similarities)
        
        if max_similarity < 0.2: # Threshold for unknown queries
            return "I'm sorry, I don't have information on that topic. You can ask me about admissions, fees, courses, or hostels!", self.get_sentiment(user_input)
        
        best_match_idx = np.argmax(similarities)
        tag = self.pattern_to_tag[best_match_idx]
        
        # Select a random response from the matched intent
        for intent in self.intents:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return response, self.get_sentiment(user_input)

if __name__ == "__main__":
    # Quick test
    bot = Chatbot("data.json")
    test_queries = ["Hi", "How to get admission?", "Tell me about fees", "What about deep learning?", "Thanks!"]
    
    for query in test_queries:
        response, sentiment = bot.get_response(query)
        print(f"User: {query}")
        print(f"Bot: {response} (Sentiment: {sentiment})\n")
