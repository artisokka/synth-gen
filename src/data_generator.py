import pandas as pd
import random
import re
import ollama

# Load the diabetes dataset
df = pd.read_csv("../data/diabetes.csv")

class DataGenerator:
    def __init__(self):
        # TODO: Default values for now
        self.features = {
            'age': 28,  # Default age for now
            'pregnancies': 2,  # Default pregnancies for now
            'insulin': 'slightly elevated',  # Default insulin level for now
        }

    def extract_features(query):
        """Use Ollama to extract relevant features from the user's query"""
        # Load the TinyLlama model via Ollama
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": query}])
        
        # Print the response to inspect the structure
        print("Response:", response)
        
        # Extract the content of the response
        response_content = response['message']['content']  # Adjusted based on the response structure
        
        # Initialize default values
        features = {
            'age': 28,  # Default age for now
            'pregnancies': 2,  # Default pregnancies for now
            'insulin': 'slightly elevated',  # Default insulin level for now
        }
        
        # Use regular expressions to extract age, pregnancies, and insulin level from the response
        age_match = re.search(r'\b(\d{2})\b(?:[^\d]*?year-old)', response_content)
        pregnancies_match = re.search(r'(\d+)\s*pregnancy', response_content, re.IGNORECASE)
        insulin_match = re.search(r'insulin level[^.]*?(\d+\.\d+|\d+)[^0-9]*ng/mL', response_content)
        
        # Update features if matches are found
        if age_match:
            features['age'] = int(age_match.group(1))
        if pregnancies_match:
            features['pregnancies'] = int(pregnancies_match.group(1))
        if insulin_match:
            features['insulin'] = insulin_match.group(1)
        
        return features
    
    # TODO: Synthetic data gen
