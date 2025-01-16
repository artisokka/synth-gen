# WIP: Synthetic Data Generator Tool

This project generates synthetic data for diabetes research using a natural language query. The user provides a query (e.g., "Generate data for a 28-year-old woman with two pregnancies and slightly elevated insulin levels"), and the app uses machine learning techniques to generate synthetic diabetes data based on the provided features.

## Features

- Extracts relevant features from natural language queries
- Generates synthetic diabetes data based on real-world statistical characteristics.
- Supports querying for individuals with varying health parameters.

## Requirements

To run this project, you need to install the required Python packages, included in requirements.txt

### Required Packages:

- pandas==2.2.3
- click==8.1.8
- ollama==0.4.6
- numpy==2.2.1

### Installation Steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/artisokka/synth-gen.git
    ```

2. Navigate to the project folder:

    ```bash
    cd synth-gen
    ```

3. Install the required packages using `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can run the program via the command line. Use the following command to generate synthetic data:

```bash
python main.py "Generate data for a 28-year-old woman with two pregnancies and slightly elevated insulin levels."
