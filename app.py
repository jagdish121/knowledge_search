import streamlit as st
from utils import (extract_text_from_pdf, preprocess_text, chunk_text, 
                   embed_texts, initialize_qdrant, upload_to_qdrant, search_qdrant)
import os
import requests

# Groq API settings
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = st.secrets.get("QDRANT_CLUSTER_URL") or os.getenv("QDRANT_CLUSTER_URL")

# Initialize DB
COLLECTION_NAME = "pdf_chunks"
client = initialize_qdrant(COLLECTION_NAME, QDRANT_CLUSTER_URL, QDRANT_API_KEY)

st.title("📚 Knowledge Search System")
st.write("Upload your PDF, and ask anything!")

# Sidebar: Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = preprocess_text(raw_text)
        chunks = chunk_text(cleaned_text)
        embeddings = embed_texts(chunks)
        upload_to_qdrant(client, COLLECTION_NAME, chunks, embeddings)
    st.success("PDF Uploaded and Processed!")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file:
    prompt = st.chat_input("Ask something about your PDF...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Search relevant chunks
        context_chunks = search_qdrant(client, COLLECTION_NAME, prompt)
        context_text = "\n\n".join(context_chunks)

        # Call LLaMA 4 (Groq API)
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # or another model you enabled
            "messages": [
                {"role": "system", "content": "Use the following context to answer questions."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{prompt}"}
            ],
            "temperature": 0.2
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        result = response.json()

        # Initialize reply with a default value
        reply = "Sorry, I couldn't find any response."

        if 'choices' in result:
            reply = result['choices'][0]['message']['content']
        else:
            st.error("❌ Error: No choices found in response.")

        with st.chat_message("assistant"):
            st.markdown(reply)

        # Append the reply to session state messages
        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("Upload a PDF first!")

