import pandas as pd
import random
import re
import ollama

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
        prompt = f"This is the query: \n{response_content}\n\nExamine it and succinctly extract the following features from it if present: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, outcome. If not present, enter None"
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        
        # Extract the parsed features from the response
        parsed_query = response['message']['content']

        print(parsed_query) # debug
        
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

        print(parsed_features)  # debug

        return parsed_features


    def generate_synthetic_data(self, features):
        """Generate synthetic data based on extracted features"""
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
            filtered_df = filtered_df[filtered_df['Age'] == int(age)]
        if pregnancies is not None:
            filtered_df = filtered_df[filtered_df['Pregnancies'] == int(pregnancies)]
        if glucose is not None:
            filtered_df = filtered_df[filtered_df['Glucose'] == int(glucose)]
        if blood_pressure is not None:
            filtered_df = filtered_df[filtered_df['BloodPressure'] == int(blood_pressure)]
        if bmi is not None:
            filtered_df = filtered_df[filtered_df['BMI'] == float(bmi)]
        if diabetes_pedigree_function is not None:
            filtered_df = filtered_df[filtered_df['DiabetesPedigreeFunction'] == float(diabetes_pedigree_function)]
        
        # If no matching data found, sample from the entire dataset
        if filtered_df.empty:
            print(f"No matching data for the specified features, sampling from the entire dataset.")
            filtered_df = df.sample(n=1)

        # Generate synthetic data for missing features based on statistics of the filtered group
        # Use the mean or median of the filtered data for missing values
        if insulin_level is None:
            insulin = filtered_df['Insulin'].mean()  # Use the mean if no insulin level is provided
        else:
            insulin = insulin_level  # Use the provided insulin level

        # If any of the other features are missing, use the statistical mean or median for the filtered group
        if age is None:
            age = filtered_df['Age'].mean()  # Use the mean age in the filtered data
        if pregnancies is None:
            pregnancies = filtered_df['Pregnancies'].mode()[0]  # Use the mode (most frequent) pregnancies value
        if glucose is None:
            glucose = filtered_df['Glucose'].mean()  # Use the mean glucose level
        if blood_pressure is None:
            blood_pressure = filtered_df['BloodPressure'].mean()  # Use the mean blood pressure
        if bmi is None:
            bmi = filtered_df['BMI'].mean()  # Use the mean BMI
        if diabetes_pedigree_function is None:
            diabetes_pedigree_function = filtered_df['DiabetesPedigreeFunction'].mean()  # Use the mean value

        # Sample the remaining features from the filtered dataset
        sample = filtered_df.sample(n=1)

        # Return the generated synthetic data, with statistical generation where applicable
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
