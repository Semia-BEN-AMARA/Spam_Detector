import pandas as pd
from transformers import AutoTokenizer

def load_raw_data(file_path):
    """
    Loads the raw spam dataset using latin-1 encoding and safely handles column mapping.
    """
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # Security: Map legacy columns 'v1' and 'v2' to explicit names if present
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    
    # Strict validation of required structural columns
    if 'label' not in df.columns or 'text' not in df.columns:
        raise KeyError(f"Columns 'label' and 'text' not found. Available columns: {list(df.columns)}")
        
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    return df['text'].values, df['label_encoded'].values

def create_bert_tokenizer():
    """
    Instantiates the official pre-trained DistilBERT tokenizer.
    """
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

def prepare_inputs_for_bert(texts, tokenizer, max_len=120):
    """
    Tokenizes raw text strings into native DistilBERT format (input_ids and attention_mask).
    """
    tokenized = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }