import streamlit as st
import os
from utils.preprocessing import load_artifacts, normalize_query_text
from utils.inference import Recommender
from utils.ui import render_recommendation_card, copy_button

st.set_page_config(page_title="CNN-RECOMMENDER", layout="wide")
st.title("Retail Smart Recommender – CNN-RECOMMENDER")
st.markdown("**Sistem rekomendasi dua arah untuk produk HijrahFood (Masakan → Produk / Produk → Masakan)**")

# ---------- LOAD ARTIFACTS ----------
ART_PATH_CANDIDATES = ['./artifacts', '/content/artifacts', '/app/artifacts']
art_path = None

for p in ART_PATH_CANDIDATES:
    if os.path.exists(p):
        art_path = p
        break

if art_path is None:
    st.error("❌ Folder artifacts/ tidak ditemukan. Upload tokenizer.pkl, sku_embeddings.pkl, textcnn_model.pt, products.csv.")
    st.stop()

tokenizer, sku_list, product_embeddings, model = load_artifacts(art_path)

recommender = Recommender(
    tokenizer=tokenizer,
    sku_list=sku_list,
    product_embeddings=product_embeddings,
    model=model
)

# ---------- SIDEBAR ----------
st.sidebar.header("Pengaturan")
top_k = st.sidebar.slider("Jumlah rekomendasi (Top-K)", 1, 12, 5)
mode = st.sidebar.selectbox("Mode rekomendasi", ["Masakan → Produk", "Produk → Masakan"])

# ---------- MAIN INPUT ----------
st.markdown("---")
q = st.text_input("Masukkan masakan atau nama produk (contoh: 'rendang' atau 'ayam boneless dada')")

if st.button("Cari"):
    if not q:
        st.warning("Masukkan query teks dulu.")
    else:
        seq = tokenizer.texts_to_sequences([q])
        pad = pad_sequences(seq, maxlen=28, padding='post')

        # Dapatkan query embedding
        import torch
        with torch.no_grad():
            q_emb = model(torch.tensor(pad).to(DEVICE)).cpu().numpy()[0]

        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        sims = product_embeddings_norm.dot(q_norm)

        idxs = np.argsort(sims)[::-1][:k]

        st.subheader("Hasil Rekomendasi:")

        for i in idxs:
            sku = sku_list[i]
            nama = df.loc[i, name_col]  # ambil nama produk
            score = float(sims[i])

            st.markdown(
                f"""
                <div style="padding:15px; margin:10px 0; border-radius:10px;
                            background:#f2f2f2; border:1px solid #ccc;">
                    <h4 style="margin:0;">{nama}</h4>
                    <p style="margin:5px 0;"><b>SKU:</b> {sku}</p>
                    <p style="margin:5px 0;"><b>Kecocokan:</b> {score:.3f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


