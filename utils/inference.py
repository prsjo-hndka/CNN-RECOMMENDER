import numpy as np
import torch
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Recommender:
    def __init__(self, tokenizer, sku_list, product_embeddings, model=None, maxlen=28):
        self.tokenizer = tokenizer
        self.sku_list = sku_list
        self.product_embeddings = product_embeddings
        self.model = model
        self.maxlen = maxlen

        if product_embeddings is not None:
            self.product_embeddings_norm = normalize(product_embeddings, axis=1)
        else:
            self.product_embeddings_norm = None

    def encode_query(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=self.maxlen, padding="post")

        if self.model is None:
            return None

        with torch.no_grad():
            t = torch.tensor(pad, dtype=torch.long)
            emb = self.model(t).cpu().numpy()[0]
            return emb

    def _usage_suggestions(self, sku):
        s = str(sku).lower()

        if "ayam" in s:
            return ["Cocok untuk sup / tumis", "Bisa digunakan untuk sate / goreng"]

        if "sapi" in s or "beef" in s:
            return ["Cocok untuk rendang / semur", "Ideal untuk slow-cook"]

        return ["Cocok untuk berbagai masakan umum"]

    def query(self, text, top_k=5):
        emb = self.encode_query(text)
        results = []

        if emb is not None:
            q = emb / (np.linalg.norm(emb) + 1e-9)
            sims = self.product_embeddings_norm.dot(q)
            idxs = np.argsort(sims)[::-1][:top_k]

            for i in idxs:
                results.append({
                    "SKU": self.sku_list[i],
                    "Nama Barang": "",
                    "score": float(sims[i]),
                    "usage": self._usage_suggestions(self.sku_list[i])
                })
            return results

        # fallback
        for i in range(top_k):
            results.append({
                "SKU": self.sku_list[i],
                "Nama Barang": "",
                "score": 0.0,
                "usage": self._usage_suggestions(self.sku_list[i])
            })
        return results

