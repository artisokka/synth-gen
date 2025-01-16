import click
from data_generator import DataGenerator

@click.command()
@click.argument('query')
def main(query):
    """CLI for generating synthetic diabetes data based on a natural language query."""
    # Extract features from the user's query using TinyLlama
    features = DataGenerator.extract_features(query)
    
    if not features:
        print("No valid features found in the query.")
        return
    
    print(features)

if __name__ == "__main__":
    main()