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

    def extract_features(self, query):
        """Use Ollama to extract relevant features from the user's query"""
        # Load the TinyLlama model via Ollama
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": query}])
        
        # Print the response to inspect the structure
        print("Response:", response)
        
        # Extract the content of the response
        response_content = response['message']['content']  # Adjusted based on the response structure
        
        # Initialize default values (HUOM)
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

    def generate_synthetic_data(self, features):
        """Generate synthetic data based on extracted features"""
        # Extract features
        age = features['age']
        pregnancies = features['pregnancies']
        insulin_level = features['insulin']  # 'slightly elevated', etc.

        # Filter the dataset based on age and gender
        filtered_df = df[(df['Age'] == age) & (df['Pregnancies'] == pregnancies)]

        if filtered_df.empty:
            raise ValueError(f"No data found for the given features: Age={age}, Pregnancies={pregnancies}, Gender={gender}")

        # Sample insulin value based on the filtered dataset
        if insulin_level == 'slightly elevated':
            insulin_range = filtered_df[filtered_df['Insulin'] > filtered_df['Insulin'].median()]['Insulin']
        else:
            insulin_range = filtered_df['Insulin']

        # Ensure insulin range is not empty
        if insulin_range.empty:
            insulin = filtered_df['Insulin'].median()  # Use the median if no suitable insulin level is found
        else:
            insulin = insulin_range.sample(n=1).iloc[0]  # Randomly select one value from the insulin range

        # Sample the rest of the data
        sample = filtered_df.sample(n=1)

        # Return the generated synthetic data
        return {
            'Age': age,
            'BMI': sample['BMI'].values[0],
            'DiabetesPedigreeFunction': sample['DiabetesPedigreeFunction'].values[0],
            'Glucose': sample['Glucose'].values[0],
            'BloodPressure': sample['BloodPressure'].values[0],
            'SkinThickness': sample['SkinThickness'].values[0],
            'Insulin': insulin,
            'Outcome': sample['Outcome'].values[0]
        }

