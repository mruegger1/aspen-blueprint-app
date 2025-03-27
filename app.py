import streamlit as st
import pandas as pd

st.set_page_config(page_title="Aspen Blueprint", layout="wide")

st.title("🏔️ Aspen Blueprint - Comparable Finder MVP")

uploaded_file = st.file_uploader("Upload Aspen CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample Data", df.head())

    # Filters
    bedrooms = st.selectbox("Bedrooms", sorted(df['bedrooms'].dropna().unique()))
    area = st.selectbox("Area", sorted(df['area'].dropna().unique()))

    # Filtered comps
    filtered = df[(df['bedrooms'] == bedrooms) & (df['area'] == area)]
    st.write("### Filtered Comps", filtered)
