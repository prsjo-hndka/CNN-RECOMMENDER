import os, pickle, re
import pandas as pd
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

def normalize_text(t: str) -> str:
    t = str(t).lower()
    t = t.replace('/', ' ')
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^\w\s\u00C0-\u017F]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.replace('karkas', 'ayam karkas')
    return t

def normalize_query_text(text: str) -> str:
    return normalize_text(text)

def load_artifacts(art_path: str):
    """
    Returns:
      tokenizer, sku_list, product_embeddings (np.array), model_or_None, df_products (pandas.DataFrame)
    """
    # tokenizer
    tk_path = os.path.join(art_path, "tokenizer.pkl")
    if not os.path.exists(tk_path):
        raise FileNotFoundError(f"tokenizer.pkl tidak ditemukan di {art_path}")
    with open(tk_path, "rb") as f:
        tokenizer = pickle.load(f)

    # embeddings
    emb_path = os.path.join(art_path, "sku_embeddings.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"sku_embeddings.pkl tidak ditemukan di {art_path}")
    with open(emb_path, "rb") as f:
        data = pickle.load(f)
        sku_list = data.get("sku_list", [])
        product_embeddings = data.get("embeddings", None)

    # try load optional model (define class inline)
    model = None
    model_path = os.path.join(art_path, "textcnn_model.pt")
    if os.path.exists(model_path):
        try:
            import torch.nn as nn
            class TextCNNEncoder(nn.Module):
                def __init__(self, vocab_size, embed_dim=128, out_dim=128, kernel_sizes=[2,3,4], num_filters=128, dropout=0.3):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                    self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k, padding=k//2) for k in kernel_sizes])
                    self.pool = nn.AdaptiveMaxPool1d(1)
                    self.fc = nn.Linear(num_filters * len(kernel_sizes), out_dim)
                    self.dropout = nn.Dropout(dropout)
                def forward(self, x):
                    emb = self.embedding(x).permute(0,2,1)
                    outs = []
                    for conv in self.convs:
                        c = torch.relu(conv(emb))
                        p = self.pool(c).squeeze(-1)
                        outs.append(p)
                    h = torch.cat(outs, dim=1)
                    h = self.dropout(h)
                    out = self.fc(h)
                    out = out / (out.norm(dim=1, keepdim=True) + 1e-8)
                    return out

            vocab_size = min(8000, len(tokenizer.word_index) + 1)
            model = TextCNNEncoder(vocab_size=vocab_size)
            state = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            # jika gagal load model, jangan crash — warn dan lanjut tanpa model
            print("Warning: gagal load textcnn_model.pt:", e)
            model = None

    # load products.csv (for nama lookup) — jika tidak ada, create placeholder df
    products_path = os.path.join(art_path, "products.csv")
    if os.path.exists(products_path):
        df_products = pd.read_csv(products_path)
    else:
        # buat df kecil yang hanya berisi SKU kolom sebagai fallback
        df_products = pd.DataFrame({"SKU": sku_list})

    return tokenizer, sku_list, product_embeddings, model, df_products
