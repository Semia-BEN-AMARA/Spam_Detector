# AT&T Spam Detector 🕵️‍♀️

## Project Overview
An end-to-end NLP solution developed for AT&T to detect and filter out spam SMS traffic. This repository demonstrates a clear engineering progression from a basic neural baseline up to modern Transformer Fine-Tuning (**DistilBERT**), optimized to handle severe class imbalance (~86.6% Ham / ~13.4% Spam).

## 📊 Performance Benchmark

Our main priority was to minimize **False Positives** (blocking a legitimate user message by mistake). Therefore, **Spam Precision** is our key operational metric.

| Model Architecture | Spam Precision | Spam Recall | Spam F1-Score | Operational Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **1. Baseline (Embedding + Pooling)** | 1.00* | 0.01 | 0.01 | **Failed.** High precision is an illusion; the model systematically guessed "ham". |
| **2. RNN (Bidirectional LSTM)** | 0.99 | 0.89 | 0.94 | **Excellent Alternative.** Lightweight with high sequential awareness. |
| **3. DistilBERT (Transformers)** | **0.99** | **0.96** | **0.97** | **Best Choice.** Safest production boundary (only 2 errors on the entire test set). |

## 📁 Repository Structure

* `notebooks/eda_and_modeling.ipynb` 🧪 : **The Lab Phase.** Contains Exploratory Data Analysis, model benchmarking, and performance charts.
* `src/preprocessing.py` ⚙️ : **The Production Pipeline.** Clean data loading and native DistilBERT sub-word tokenization.
* `src/model.py` 🧠 : **The Production Deployment Wrapper.** Independent structural definition and compilation of the selected DistilBERT model.
* `requirements.txt` 📌 : Environment dependencies.

## 🚀 How to Run

### 1. Installation
Clone the repository and install the required dependencies in a Python 3.10 environment:
```bash
git clone [https://github.com/Semia-BEN-AMARA/Spam_Detector.git](https://github.com/Semia-BEN-AMARA/Spam_Detector.git)
cd Spam_Detector
pip install -r requirements.txt