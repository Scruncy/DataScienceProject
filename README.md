# Predicting Claim Frequency using ML-Algorithms: A case study for Belgian Motor-TPL Insurance

This project provides the code for the report for Actuarial Data Science. It includes a modelling module (`Model_implementation.py`) for the Machine Learning models and visualizations (`Visualizations.py`) for plotting the results. To run the files, it is required to use Python 3.10

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [License](#license)
- [Authors](#authors)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Scruncy/DataScienceProject
   cd DataScienceProject
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. (Optional) Run the models script to interact with the tool:
   ```bash
   python Model_implementation.py
   ```
2. The relevant models will be stored in the model_metadata.csv file:

3. Run the Visualizations file to plot the graphs:
   ```bash
   python Visualizations.py
   ```

## Project Structure

- **`Model_implementation.py`**: This module contains all the computations for the Machine Learning Models used in the report.
  
- **`Visualizations.py`**: This script provides the code for all the visualizations in the report.

## Features

- Train different types of Machine Learning models on the beMTPL97 dataset.
- Plotting capabilities to visualize findings.

## License

This project does not currently have a license. Please contact the author if you have questions about usage.

## Authors

- **Evert Van Hecke** - [evanhecke](https://github.com/evanhecke)
- **Simone Deponte** - [Scruncy](https://github.com/evanhecke)