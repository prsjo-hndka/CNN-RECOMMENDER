import streamlit as st

def render_recommendation_card(item):
    st.markdown(
        f"""
        <div style="padding:15px; background:#fafafa; border-radius:12px;
                    border:1px solid #ddd; margin-bottom:10px;">
            <h4 style="margin:0; color:#333;">{item['Nama Barang']}</h4>
            <p style="margin:4px 0;"><b>SKU:</b> {item['SKU']}</p>
            <p style="margin:4px 0;"><b>Score:</b> {item['score']:.3f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def copy_button(text, key):
    st.button("Copy Detail", key=key, on_click=lambda: st.write(text))
