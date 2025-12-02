import os, pickle, re, numpy as np
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
    tokenizer = None
    sku_list = []
    product_embeddings = None
    model = None

    # tokenizer
    with open(os.path.join(art_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    # embeddings
    with open(os.path.join(art_path, "sku_embeddings.pkl"), "rb") as f:
        data = pickle.load(f)
        sku_list = data["sku_list"]
        product_embeddings = data["embeddings"]

    # optional model
    model_path = os.path.join(art_path, "textcnn_model.pt")
    if os.path.exists(model_path):

        import torch.nn as nn
        class TextCNNEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=128, out_dim=128,
                         kernel_sizes=[2,3,4], num_filters=128, dropout=0.3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
                    for k in kernel_sizes
                ])
                self.pool = nn.AdaptiveMaxPool1d(1)
                self.fc = nn.Linear(num_filters * len(kernel_sizes), out_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                emb = self.embedding(x).permute(0,2,1)
                outs=[]
                for conv in self.convs:
                    c = torch.relu(conv(emb))
                    p = self.pool(c).squeeze(-1)
                    outs.append(p)
                h = torch.cat(outs, dim=1)
                h = self.dropout(h)
                out = self.fc(h)
                return out / (out.norm(dim=1, keepdim=True) + 1e-8)

        vocab_size = min(8000, len(tokenizer.word_index)+1)
        model = TextCNNEncoder(vocab_size=vocab_size)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

    return tokenizer, sku_list, product_embeddings, model

