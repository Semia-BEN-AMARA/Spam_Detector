import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

def build_production_bert_model():
    """
    Instantiates and compiles the pre-trained DistilBERT model 
    with production configurations for AT&T sequence classification.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=2
    )
    
    # Standard production compilation mapping cross-entropy loss from raw logits
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Injecting the highly stable state-of-the-art transformer learning rate
    tf.keras.backend.set_value(model.optimizer.learning_rate, 3e-5)
    
    return model