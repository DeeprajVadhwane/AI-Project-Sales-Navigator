import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoModel, AutoTokenizer

# Load your product dataset (with columns like 'product_name', 'product_description', 'product_price')
@st.cache_data
def load_data():
    data = pd.read_csv("products_sample.csv")
    data.dropna(subset=['product_name', 'product_description', 'product_price'], inplace=True)
    return data

@st.cache_resource
def load_model():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


st.title("🛍️ Product Search Chatbot")

data = load_data()
tokenizer, model = load_model()

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = np.vstack(data.apply(
        lambda row: generate_embedding(row['product_name'] + " " + row['product_description'], tokenizer, model), 
        axis=1).values
    )
embedding_dim = st.session_state['embeddings'].shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(st.session_state['embeddings'])

if 'user_memory' not in st.session_state:
    st.session_state['user_memory'] = []

def search_products(query, k=5):
    query_embedding = generate_embedding(query, tokenizer, model)
    distances, indices = index.search(np.array([query_embedding]), k)
    
    similarities = [cosine_similarity(query_embedding, st.session_state['embeddings'][i]) for i in indices[0]]
    
    results = data.iloc[indices[0]].copy()
    results['similarity'] = similarities
    return results.sort_values(by='similarity', ascending=False)

user_input = st.text_input("🔍 Ask me about a product:", placeholder="Type your query here...")

if st.button("Search"):
    if user_input:
        results = search_products(user_input)
        st.header("Search Results:")
        
        if results.empty:
            st.warning("No products found matching your query.")
        else:
            
            cols = st.columns(3)  
            for idx, row in results.iterrows():
                with cols[idx % 3]:
                    st.markdown(
                        f"""
                        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 10px; margin: 10px;'>
                            <h4 style='color: #333;'>{row['product_name']}</h4>
                            <p>{row['product_description']}</p>
                            <p style='font-weight: bold;'>Price: ${row['product_price']:.2f}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        st.session_state['user_memory'].append(user_input)
    else:
        st.warning("Please enter a product name or description to search.")
