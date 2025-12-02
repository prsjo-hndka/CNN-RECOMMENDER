import numpy as np
import torch
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Recommender:
    def __init__(self, tokenizer, sku_list, product_embeddings, model=None, df_products=None, maxlen=28):
        self.tokenizer = tokenizer
        self.sku_list = sku_list
        self.product_embeddings = product_embeddings
        self.model = model
        self.df = df_products
        self.maxlen = maxlen
        if self.product_embeddings is not None:
            self.product_embeddings_norm = normalize(self.product_embeddings, axis=1)
        else:
            self.product_embeddings_norm = None

    def encode_query(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=self.maxlen, padding='post')
        if self.model is None:
            return None
        with torch.no_grad():
            t = torch.tensor(pad, dtype=torch.long)
            emb = self.model(t).cpu().numpy()[0]
        return emb

    def _usage_suggestions(self, sku):
        # very simple rule-based usage by sku string or product name
        s = ''
        # try find name in df
        try:
            idx = self.sku_list.index(sku)
            if self.df is not None and idx < len(self.df):
                s = str(self.df.iloc[idx].iloc[0]).lower()
        except Exception:
            s = str(sku).lower()

        suggestions = []
        if 'ayam' in s or 'chicken' in s:
            suggestions = ['Cocok untuk sup atau tumis', 'Bisa untuk sate atau goreng']
        elif 'sapi' in s or 'beef' in s or 'daging' in s:
            suggestions = ['Cocok untuk rendang dan semur', 'Baik untuk stew dan slow-cook']
        else:
            suggestions = ['Cocok untuk berbagai masakan rumah tangga']
        return suggestions

    def query(self, text, top_k=5):
        emb = self.encode_query(text)
        results = []
        if emb is not None and self.product_embeddings_norm is not None:
            q = emb / (np.linalg.norm(emb) + 1e-9)
            sims = self.product_embeddings_norm.dot(q)
            idxs = np.argsort(sims)[::-1][:top_k]
            for i in idxs:
                sku = self.sku_list[i]
                # get product name from df if possible
                name = None
                try:
                    if self.df is not None and i < len(self.df):
                        # try common name column
                        cols = [c for c in self.df.columns if 'nama' in c.lower() or 'name' in c.lower()]
                        if len(cols) > 0:
                            name = str(self.df.iloc[i][cols[0]])
                        else:
                            name = str(self.df.iloc[i].iloc[0])
                except Exception:
                    name = ''
                results.append({
                    'SKU': sku,
                    'Nama Barang': name if name is not None else str(sku),
                    'score': float(sims[i]),
                    'usage': self._usage_suggestions(sku)
                })
            return results

        # fallback if model not available: return top-k by index with zero score
        for i in range(min(top_k, len(self.sku_list))):
            sku = self.sku_list[i]
            name = ''
            try:
                if self.df is not None and i < len(self.df):
                    cols = [c for c in self.df.columns if 'nama' in c.lower() or 'name' in c.lower()]
                    if len(cols) > 0:
                        name = str(self.df.iloc[i][cols[0]])
                    else:
                        name = str(self.df.iloc[i].iloc[0])
            except Exception:
                name = ''
            results.append({'SKU': sku, 'Nama Barang': name if name else str(sku), 'score': 0.0, 'usage': self._usage_suggestions(sku)})
        return results
