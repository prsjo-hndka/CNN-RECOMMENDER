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
    if not q.strip():
        st.warning("⚠️ Masukkan query terlebih dahulu.")
    else:
        query = normalize_query_text(q)
        results = recommender.query(query, top_k=top_k)
        st.success(f"Menampilkan {len(results)} rekomendasi untuk: **{q}**")

        cols_per_row = 2
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, r in enumerate(results[i:i+cols_per_row]):
                with cols[j]:
                    render_recommendation_card(r)
                    summary = (
                        f"SKU: {r['SKU']}\n"
                        f"Nama: {r['Nama Barang']}\n"
                        f"Skor: {r['score']:.3f}\n"
                        f"Rekomendasi penggunaan:\n- " + "\n- ".join(r.get('usage', []))
                    )
                    copy_button(summary, key=f"copy_{i+j}")

