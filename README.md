# AT&T Spam Detector 🕵️‍♀️

## Project Overview
This repository contains a Deep Learning solution developed for AT&T to automate the detection of spam SMS messages. The goal is to build an end-to-end text classification pipeline capable of accurately filtering out spam based solely on message content, protecting users from unwanted exposure.

## Dataset
The project utilizes an SMS Spam collection dataset consisting of text messages tagged as either `ham` (legitimate) or `spam`. 

## Project Structure
* `data/`: Contains raw and processed datasets (excluded from version control).
* `notebooks/`: Jupyter notebooks used for Exploratory Data Analysis (EDA), data preprocessing, and model training.
* `src/`: Reusable production-ready Python modules for text processing and modeling.

## Approach & Methodology
1. **Exploratory Data Analysis (EDA):** Understanding class imbalance, sentence length distributions, and word frequencies.
2. **Preprocessing:** Text cleaning, tokenization, padding, and text-to-sequence vectorization.
3. **Modeling:** * Baseline Deep Learning architectures (Embedding + GlobalAveragePooling / Simple RNN / LSTM).
   * Transfer Learning using pre-trained word embeddings or advanced language models if necessary.
4. **Evaluation:** Assessing performance using precision, recall, and F1-score to handle class imbalance effectively.

## How to Run
1. Clone this repository:
   ```bash
   git clone [https://github.com/Semia-BEN-AMARA/Spam_Detector.git]