# Author Classification with RNN Language Models

This project implements an author classification system using recurrent neural networks (RNNs) to analyze sentences from four classic literary texts. The model is trained to predict which author is most likely to have written a given sentence by calculating probabilities based on trained language models.

---

## Table of Contents
- [Description](#description)
- [Dataset](#dataset)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

---

## Description

The system performs the following tasks:
1. Processes four books by splitting text into sentences and building a vocabulary of unique words.
2. Partitions the dataset into training and testing sets (80% training, 20% testing).
3. Trains separate RNN models for each book.
4. Tests each model on sentences from all books to classify their authorship using Bayesian probability.

---

## Dataset

The dataset comprises four shuffled text files derived from classic novels:
- `Wuthering Heights`
- `A Room with a View`
- `Great Expectations`
- `Emma`

File names:
- `wuthering_sentences_shuf_6379`
- `room_with_a_sentences_shuf_6379`
- `great_expectations_sentences_shuf_6379`
- `emma_sentences_shuf_6379`

These files should be placed in the same directory as the code.

---

## Setup and Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- PyTorch
- NumPy
- Additional Python libraries: `re`, `json`, `collections`, `os`, `random`

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/briebi/author-prediction-rnn.git
    cd authot-prediction-rnn
    ```
2. Install the required Python libraries:
    ```bash
    pip install torch
    ```

---

## Usage

### Running the Code
1. Ensure the shuffled text files are in the same directory as the script.
2. Execute the Python script:
    ```bash
    python main.py
    ```

### Output
The script will:
- Print the total number of words and sample vocabulary.
- Train models for each book, displaying epoch losses.
- Evaluate test sentences and output the average probabilities for each model.

---

## Code Structure

### Key Components
- **Data Preprocessing**:
  - Reads and tokenizes text into sentences and words.
  - Builds a master vocabulary and enumerates words.
  - Partitions sentences into training and testing sets.

- **Training**:
  - Each book has its own RNN language model.
  - Uses CrossEntropy loss and Adam optimizer.
  - Prints epoch-wise losses during training.

- **Testing**:
  - Models evaluate sentences from all books.
  - Calculates probabilities using Bayesian principles.

### File Dependencies
- Python script for the model (`<script-name>.py`).
- Text files (`wuthering_sentences_shuf_6379`, etc.).

---

## Model Architecture

Each RNN model consists of:
- An embedding layer for word representation (`d_emb = 128`).
- A single-layer LSTM network (`num_layers = 1`, `d_hid = 128`).
- A linear layer for output classification.

Training parameters:
- Learning rate: `0.0003`
- Epochs: `5`
- Loss function: CrossEntropyLoss

---

## Results

The system predicts the likelihood of a sentence being authored by each of the four authors using trained RNN models. The output includes:
- Average probabilities for each book.
- Metrics comparing models' performance.
