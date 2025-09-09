# Dynamic Knowledge Base Chatbot with Groq SDK

This project is a simple yet powerful chatbot that uses the Groq LLM, a FAISS vector store, and a Streamlit-based interface. The chatbot can answer user questions by searching a knowledge base and using Groq's large language model to generate intelligent responses.

---

## Overview

The system works in the following way:

1. The user asks a question in the Streamlit app.
2. The system performs a similarity search in the FAISS vector database to retrieve relevant context.
3. The question and context are sent to the Groq LLM API, which returns a helpful answer.

Additionally, the vector database is kept up to date by periodically fetching new content from the web (for example, Wikipedia articles).

---

## Project Structure

### app.py

This is the main Streamlit application.

* Allows the user to input a question.
* Retrieves the most relevant documents from the vector store.
* Uses the Groq API to generate a natural language answer.

### scheduled\_update.py

This script keeps the vector database updated automatically.

* It runs every 6 hours.
* Fetches the latest content from a specified URL and updates the FAISS database.

### utils.py

Helper functions for:

* Downloading and processing web page content.
* Splitting large texts into manageable chunks.
* Updating the FAISS vector store by adding new document embeddings.

---

## Prerequisites

Make sure you have Python 3.8 or higher installed.

Install the required dependencies using pip:

```bash
pip install streamlit requests beautifulsoup4 langchain huggingface_hub faiss-cpu schedule python-dotenv
```

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key
```

---

## How to Use

### Step 1: Initialize the Vector Database

Before running the chatbot, you should populate the FAISS vector store at least once.
Example (from a Python shell):

```python
from utils import update_vector_db
update_vector_db("https://en.wikipedia.org/wiki/Artificial_intelligence")
```

### Step 2: Run the Chatbot Application

```bash
streamlit run app.py
```

Open your browser and visit `http://localhost:8501` to interact with the chatbot.

### Step 3: Keep Vector Store Updated Automatically

```bash
python scheduled_update.py
```

This will continuously run and update the vector database every 6 hours using the specified webpage URL.

---

## Configuration Details

* Embeddings Model:
  `"sentence-transformers/all-MiniLM-L6-v2"`

* Groq LLM Model:
  `"llama-3.3-70b-versatile"`

* Vector Store Path:
  Default is `"vectorstore"` folder.

---

## Things to Keep in Mind

* Make sure the FAISS database folder has enough storage space.
* Watch out for API usage limits of the Groq service.
* You can change the scheduled update URL in `scheduled_update.py` to point to any webpage of your choice.

---

## Future Improvements

* Add user-friendly error handling in the UI.
* Create an admin panel to trigger manual updates.
* Allow multiple URLs or content sources for richer knowledge.

---