# Disease Condition Prediction Based on Drug Reviews

This project aims to classify patient conditions based on drug reviews using Natural Language Processing (NLP) and Machine Learning techniques.

## Table of Contents
- [Description](#description)
- [Tech Stack and Concepts](#tech-stack-and-concepts)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Description

The project involves analyzing drug reviews to predict the conditions they are meant to treat. This involves data preprocessing, feature extraction using TF-IDF and Bag of Words, and implementing machine learning models like Naive Bayes and Passive Aggressive Classifier.

## Tech Stack and Concepts

### Tech Stack
- **Programming Languages**: Python
- **Libraries and Frameworks**: 
  - pandas
  - numpy
  - seaborn
  - scikit-learn
  - matplotlib
  - nltk
  - bs4 (BeautifulSoup)

### Concepts
- **Natural Language Processing (NLP)**
  - Text preprocessing
  - Tokenization
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Bag of Words
- **Machine Learning**
  - Naive Bayes Classifier
  - Passive Aggressive Classifier
  - Model evaluation using metrics like accuracy, precision, recall, and F1-score

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd Diagnostic-Classification-from-Pharmacological-Reviews
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook Diagnostic-Classification-from-Pharmacological-Reviews.ipynb
    ```
2. Execute the cells sequentially to run the project.

## Project Structure

- `Diagnostic-Classification-from-Pharmacological-Reviews.ipynb`: The main Jupyter Notebook containing the project code and explanations.
- `data/`: Directory containing the dataset (if applicable).
- `models/`: Directory containing saved models (if applicable).

## Dependencies

- pandas
- itertools
- string
- numpy
- seaborn
- scikit-learn
- matplotlib
- nltk
- bs4

You can install the dependencies using:
```sh
pip install pandas itertools numpy seaborn scikit-learn matplotlib nltk beautifulsoup4
