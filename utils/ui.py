import streamlit as st

def render_recommendation_card(r: dict):
    st.markdown(f"### {r.get('Nama Barang', '-')}")
    st.write(f"**SKU:** `{r['SKU']}`")
    st.write(f"**Relevansi:** {r['score']:.3f}")

    st.write("**Penggunaan umum:**")
    for u in r.get("usage", []):
        st.write(f"- {u}")

    st.markdown("---")

def copy_button(text: str, key: str):
    escaped = text.replace("\n", "\\n").replace("'", "\\'")
    js = f"""
    <script>
    function copyToClipboard_{key}() {{
        navigator.clipboard.writeText('{escaped}').then(
            () => alert('Rekomendasi disalin!'),
            () => alert('Gagal menyalin.')
        );
    }}
    </script>
    <button onclick="copyToClipboard_{key}()">Copy</button>
    """
    st.markdown(js, unsafe_allow_html=True)

