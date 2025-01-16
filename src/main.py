import argparse
import pandas as pd
from data_generator import DataGenerator

# This app generates synthetic data based on a dataset and user prompt
def main():
    print("Synthetic Data Generator")

    parser = argparse.ArgumentParser(description='Synthetic Data Generator for Medical Training Scenarios')
    parser.add_argument('--query', type=str, required=True, help='Description of the synthetic data to generate')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of synthetic data samples to generate')

    args = parser.parse_args()

    # Load the diabetes dataset
    diabetes_data = pd.read_csv('../data/diabetes.csv')

    # Initialize the data generator with the DataFrame
    data_generator = DataGenerator(diabetes_data)

    # Generate synthetic data based on the user query
    synthetic_data = data_generator.generate_synthetic_data(args.num_samples)

    # Output the generated synthetic data
    print(synthetic_data)

if __name__ == "__main__":
    main()