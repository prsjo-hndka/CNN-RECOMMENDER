import numpy as np
import torch
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Recommender:
    def __init__(self, tokenizer, sku_list, product_embeddings, model, df_products):
        self.tokenizer = tokenizer
        self.sku_list = sku_list
        self.product_embeddings = normalize(product_embeddings, axis=1)
        self.model = model
        self.df = df_products

    def embed_query(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=28, padding="post")
        t = torch.tensor(pad, dtype=torch.long)

        with torch.no_grad():
            emb = self.model(t).cpu().numpy()[0]

        return emb / (np.linalg.norm(emb) + 1e-9)

    def query(self, text, top_k=5):
        q_emb = self.embed_query(text)
        sims = self.product_embeddings.dot(q_emb)

        idxs = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in idxs:
            sku = self.sku_list[i]
            nama = str(self.df.loc[i, self.df.columns[self.df.columns.str.contains("nama|Nama", case=False)][0]])

            results.append({
                "SKU": sku,
                "Nama Barang": nama,
                "score": float(sims[i]),
                "usage": [
                    "Cocok untuk masakan sehari-hari",
                    "Dapat digunakan sebagai bahan utama menu populer"
                ]
            })

        return results
