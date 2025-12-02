import os
import pickle
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.preprocessing.text import Tokenizer

def load_artifacts(art_path):
    # Load tokenizer
    with open(os.path.join(art_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    # Load embeddings
    with open(os.path.join(art_path, "sku_embeddings.pkl"), "rb") as f:
        data = pickle.load(f)
        sku_list = data["sku_list"]
        product_embeddings = data["embeddings"]

    # Load model
    from model.textcnn import TextCNNEncoder
    model = TextCNNEncoder(vocab_size=8000, embed_dim=128, out_dim=128)
    model.load_state_dict(torch.load(os.path.join(art_path, "textcnn_model.pt"), map_location="cpu"))
    model.eval()

    # Load product table (for name lookup)
    df = pd.read_csv(os.path.join(art_path, "products.csv"))

    return tokenizer, sku_list, product_embeddings, model, df


def normalize_query_text(t):
    return t.lower().strip()
