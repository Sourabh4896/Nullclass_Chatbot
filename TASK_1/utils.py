# utils.py
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_web_content(url):
    """Fetch text from a webpage"""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = [p.get_text() for p in soup.find_all("p")]
    return " ".join(paragraphs)

def update_vector_db(url, db_path="vectorstore"):
    """Update vector DB with new content"""
    text = load_web_content(url)
    new_doc = [Document(page_content=text)]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(new_doc)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    db.add_documents(chunks)
    db.save_local(db_path)
    print("✅ Vector DB updated with new content!")
