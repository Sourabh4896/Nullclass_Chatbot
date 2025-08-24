# app.py
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def ask_groq(question, context):
    """Send query + retrieved context to Groq LLM"""
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        temperature=0.3,
        max_tokens=512
    )
    return completion.choices[0].message.content

# Streamlit UI
st.title("📚 Dynamic Knowledge Base Chatbot (Groq SDK)")
query = st.text_input("Ask a question:")

if query:
    # Retrieve context from FAISS
    docs = db.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])

    # Ask Groq
    answer = ask_groq(query, context)
    st.write("**Answer:**", answer)
