import pandas as pd
import numpy as np

# Class to generate synthetic data from a DataFrame
class DataGenerator:
    def __init__(self, data):
        self.data = data
        self.columns = self.data.columns

    def generate_synthetic_data(self, num_samples=1):
        synthetic_data = pd.DataFrame(columns=self.columns)
        
        for _ in range(num_samples):
            sample = {}
            for column in self.columns:
                if self.data[column].dtype == 'object':
                    sample[column] = np.random.choice(self.data[column].unique())
                else:
                    mean = self.data[column].mean()
                    std_dev = self.data[column].std()
                    sample[column] = np.random.normal(mean, std_dev)
            synthetic_data = synthetic_data.append(sample, ignore_index=True)
        
        return synthetic_data

def main():
    data = pd.read_csv('data/diabetes.csv')
    generator = DataGenerator(data)
    synthetic_data = generator.generate_synthetic_data(num_samples=10)
    print(synthetic_data)

if __name__ == "__main__":
    main()