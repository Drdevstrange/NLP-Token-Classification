import gradio as gr
import numpy as np
import pickle
import datetime
from keras.models import load_model

model = load_model("bilstm_model.h5")

with open("tag_mappings.pkl", "rb") as f:
    tag2idx, idx2tag = pickle.load(f)

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

max_len = 128

def preprocess_text(text):
    # Split into tokens more carefully, handling punctuation
    tokens = []
    for word in text.strip().split():
        # Handle potential abbreviations carefully
        if word.isupper() and len(word) >= 2:  # Likely an abbreviation
            tokens.append(word)
        else:
            # Split on punctuation but keep abbreviations together
            current_token = ""
            for char in word:
                if char.isalnum() or char in ['-', '_']:
                    current_token += char
                elif current_token:
                    tokens.append(current_token)
                    current_token = ""
            if current_token:
                tokens.append(current_token)
    return tokens

def encode_input(text):
    tokens = preprocess_text(text)
    # Convert tokens to lowercase unless they're potential abbreviations
    input_ids = []
    for token in tokens:
        if token.isupper() and len(token) >= 2:  # Keep abbreviations as is
            token_id = word2idx.get(token, word2idx.get("UNK", 1))
        else:
            token_id = word2idx.get(token.lower(), word2idx.get("UNK", 1))
        input_ids.append(token_id)
    
    # Padding
    padded = np.zeros((1, max_len), dtype=np.int32)
    padded[0, :len(input_ids)] = input_ids[:max_len]
    return padded, tokens

def predict(text):
    x, tokens = encode_input(text)
    if not tokens:  # Handle empty input
        return []
    
    y_probs = model.predict(x, verbose=0)
    y_pred = np.argmax(y_probs[0], axis=-1)[:len(tokens)]
    
    # Post-process predictions
    pred_tags = []
    for i, (token, pred) in enumerate(zip(tokens, y_pred)):
        tag = idx2tag[pred]
        # Apply some rules to improve prediction
        if token.isupper() and len(token) >= 2:  # Likely an abbreviation
            if tag == 'O':  # If model didn't catch it
                tag = 'B-AC'  # Mark as abbreviation
        elif i > 0 and pred_tags[-1].startswith('B-LF'):  # Previous token starts a long form
            if tag == 'O':  # If model didn't catch it
                tag = 'I-LF'  # Continue the long form
        pred_tags.append(tag)
    
    log_interaction(text, tokens, pred_tags)
    return list(zip(tokens, pred_tags))

def log_interaction(text, tokens, tags):
    entry = {
    "timestamp": datetime.datetime.now().isoformat(),
    "input_text": text,
    "tokens": tokens,
    "predicted_tags": tags
    }
    with open("inference_log.jsonl", "a") as f:
        f.write(str(entry) + "\n")


iface = gr.Interface(
fn=predict,
inputs=gr.Textbox(lines=3, placeholder="Enter a sentence..."),
outputs=gr.Dataframe(headers=["Token", "Predicted Tag"]),
title="BiLSTM Token Classifier",
description="Enter a sentence to detect abbreviation and long forms."
)

iface.launch(share=True)
