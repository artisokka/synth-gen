import pandas as pd
import random
import re
import ollama
import numpy as np

# Load the diabetes dataset
df = pd.read_csv("../data/diabetes.csv")

class DataGenerator:
    def __init__(self):
        self.features = {
            'pregnancies': None,
            'glucose': None,
            'blood_pressure': None,
            'skin_thickness': None,
            'insulin': None,
            'bmi': None,
            'diabetes_pedigree_function': None,
            'age': None,
            'outcome': None
        }

    def extract_features(self, query):
        """Use TinyLlama to extract relevant features from the user's query"""
        
        # Initialize the features dictionary with None for undefined values
        features = self.features.copy()

        # Use TinyLlama to parse the query and provide relevant feature values
        parsed_query = self.parse_with_tinyllama(query)

        # Extract parsed values from the response and update features
        for feature, value in parsed_query.items():
            if value is not None:
                features[feature] = value

        return features

    def parse_with_tinyllama(self, response_content):
        """Use TinyLlama to parse the response and return the extracted features."""
        # Prompt TinyLlama to interpret the content and return features
        prompt = f"""
        Examine the following text: "{response_content}"

        Extract the following features:
        - Pregnancies (an integer number, or None if not mentioned)
        - Glucose (an integer number, or None if not mentioned)
        - Blood Pressure (an integer number, or None if not mentioned)
        - Skin Thickness (an integer or float number, or None if not mentioned)
        - Insulin (an integer or float number, or None if not mentioned)
        - BMI (a float number, or None if not mentioned)
        - Age (an integer number, or None if not mentioned)

        For each feature, if the feature is mentioned in the text, return its value. If it is not mentioned, return `None`. Respond in this exact format:
        pregnancies: result glucose: result blood_pressure: result skin_thickness: result insulin: result bmi: result age: result
        """
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
                
        # Extract the parsed features from the response
        parsed_query = response['message']['content']

        # print(parsed_query) # debug
        
        return self.parse_data(parsed_query)

    def parse_data(self, parsed_data):
        """Parse the response from TinyLlama into a dictionary of features."""
        parsed_features = {}

        lines = parsed_data.splitlines()

        for line in lines:
            match = re.match(r'(\w+)\s*[:=]\s*(\S+)', line.strip())
            if match:
                feature_name = match.group(1).lower()
                feature_value = match.group(2)
                parsed_features[feature_name] = feature_value

        # print(parsed_features)  # debug

        return parsed_features

    def generate_synthetic_data(self, features):
        """Generate synthetic data based on extracted features, using statistical distributions"""
        # Extract features
        age = features.get('age', None)
        pregnancies = features.get('pregnancies', None)
        insulin_level = features.get('insulin', None)
        glucose = features.get('glucose', None)
        blood_pressure = features.get('blood_pressure', None)
        bmi = features.get('bmi', None)
        diabetes_pedigree_function = features.get('diabetes_pedigree_function', None)

        # Filter the dataset based on the known features (those that are not None)
        filtered_df = df
        if age is not None:
            filtered_df = filtered_df[filtered_df['Age'] == age]
        if pregnancies is not None:
            filtered_df = filtered_df[filtered_df['Pregnancies'] == pregnancies]
        if glucose is not None:
            filtered_df = filtered_df[filtered_df['Glucose'] == glucose]
        if bmi is not None:
            filtered_df = filtered_df[filtered_df['BMI'] == bmi]

        # If no matching data found, sample from the entire dataset
        if filtered_df.empty:
            print(f"No matching data for the specified features, sampling from the entire dataset.")
            filtered_df = df.sample(n=1)

        # Statistical sampling for missing features
        if insulin_level is None:
            insulin = np.random.normal(filtered_df['Insulin'].mean(), filtered_df['Insulin'].std())
        else:
            insulin = insulin_level  # Use the provided insulin level

        if age is None:
            age = np.random.normal(filtered_df['Age'].mean(), filtered_df['Age'].std())
        if pregnancies is None:
            pregnancies = np.random.choice(filtered_df['Pregnancies'].mode())
        if glucose is None:
            glucose = np.random.normal(filtered_df['Glucose'].mean(), filtered_df['Glucose'].std())
        if blood_pressure is None:
            blood_pressure = np.random.normal(filtered_df['BloodPressure'].mean(), filtered_df['BloodPressure'].std())
        if bmi is None:
            bmi = np.random.normal(filtered_df['BMI'].mean(), filtered_df['BMI'].std())
        if diabetes_pedigree_function is None:
            diabetes_pedigree_function = np.random.normal(filtered_df['DiabetesPedigreeFunction'].mean(), filtered_df['DiabetesPedigreeFunction'].std())

        # Capture correlations between features (if any)
        glucose_insulin_correlation = filtered_df[['Glucose', 'Insulin']].corr().iloc[0, 1]

        # Use the correlation between glucose and insulin to generate insulin based on glucose if required
        if glucose is not None and insulin_level is None:
            if glucose_insulin_correlation > 0:
                # Generate insulin based on glucose if they have a positive correlation
                insulin = filtered_df['Insulin'].mean() + glucose_insulin_correlation * (glucose - filtered_df['Glucose'].mean())

        # Sample the remaining features (e.g., SkinThickness, Outcome) from the filtered dataset
        sample = filtered_df.sample(n=1)

        # Return the generated synthetic data, ensuring that it follows statistical distributions
        return {
            'Age': age,
            'Pregnancies': pregnancies,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree_function,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': sample['SkinThickness'].values[0],
            'Insulin': insulin,
            'Outcome': sample['Outcome'].values[0]
        }

