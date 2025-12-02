# CNN-RECOMMENDER
Sistem rekomendasi dua arah berbasis CNN untuk produk HijrahFood.

## Cara Menjalankan
pip install -r requirements.txt
streamlit run app.py


## Cara Deploy
1. Upload semua file ke GitHub
2. Masuk Streamlit Cloud
3. Connect repository → Deploy
4. Pastikan folder `artifacts/` berisi:
   - tokenizer.pkl  
   - sku_embeddings.pkl  
   - textcnn_model.pt  
   - products.csv