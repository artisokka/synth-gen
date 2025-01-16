import click
from data_generator import DataGenerator

@click.command()
@click.argument('query')
def main(query):
    """CLI for generating synthetic diabetes data based on a natural language query."""

    # Create an instance of DataGenerator
    data_generator = DataGenerator()

    # Extract features from the user's query using TinyLlama
    features = data_generator.extract_features(query)
    
    if not features:
        print("No valid features found in the query.")
        return
    
    # Generate synthetic data based on the extracted features
    synthetic_data = data_generator.generate_synthetic_data(features)

    # Display the generated synthetic data
    print("Generated Synthetic Data:")
    for key, value in synthetic_data.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
