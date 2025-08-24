# Task 4: arXiv Expert Chatbot
import streamlit as st
import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()   

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_arxiv_sample():
    """Load sample arXiv data - replace with actual dataset loading"""
    sample_papers = [
        {
            'id': '2101.00001',
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
            'categories': 'cs.CL',
            'authors': 'Vaswani et al.'
        },
        {
            'id': '2101.00002',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'abstract': 'We introduce BERT, a new language representation model for bidirectional training.',
            'categories': 'cs.CL',
            'authors': 'Devlin et al.'
        },
        {
            'id': '2101.00003',
            'title': 'Deep Learning for Computer Vision',
            'abstract': 'This paper surveys deep learning approaches for computer vision tasks.',
            'categories': 'cs.CV',
            'authors': 'LeCun et al.'
        }
    ]
    return pd.DataFrame(sample_papers)

def search_papers(query, df, top_k=5):
    """Search papers using TF-IDF similarity"""
    # Combine title and abstract for search
    df['content'] = df['title'] + ' ' + df['abstract']
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Create corpus with user query
    corpus = df['content'].tolist() + [query]
    
    # Vectorize
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Calculate similarity
    query_vector = tfidf_matrix[-1]
    paper_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vector, paper_vectors).flatten()
    
    # Get top k papers
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            paper = df.iloc[idx]
            results.append({
                'paper': paper,
                'similarity': similarities[idx]
            })
    
    return results

def summarize_paper(paper):
    """Summarize paper using Groq"""
    prompt = f"""Provide a concise summary of this research paper:

Title: {paper['title']}
Abstract: {paper['abstract']}
Authors: {paper['authors']}
Category: {paper['categories']}

Please provide:
1. Main contribution
2. Key methodology
3. Significance in the field"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def explain_concept(concept):
    """Explain complex concepts using Groq"""
    prompt = f"""Explain the concept of "{concept}" in computer science/AI in simple terms:

1. What is it?
2. Why is it important?
3. How is it used?
4. Provide a simple example

Make the explanation accessible to someone learning the topic."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def create_category_visualization(df):
    """Create visualization of paper categories"""
    category_counts = df['categories'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Distribution of Paper Categories"
    )
    
    return fig

def main():
    st.title("📚 arXiv Expert Chatbot")
    st.write("Your AI research assistant for exploring scientific papers")
    
    # Load sample data
    df = load_arxiv_sample()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Choose mode:", [
        "Search Papers", 
        "Explain Concepts", 
        "Paper Summary",
        "Visualizations"
    ])
    
    if mode == "Search Papers":
        st.header("🔍 Search Research Papers")
        
        query = st.text_input("Enter your search query:")
        
        if st.button("Search") and query:
            results = search_papers(query, df)
            
            if results:
                st.write(f"Found {len(results)} relevant papers:")
                
                for i, result in enumerate(results):
                    paper = result['paper']
                    similarity = result['similarity']
                    
                    with st.expander(f"{i+1}. {paper['title']} (Score: {similarity:.3f})"):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Category:** {paper['categories']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.write(f"**arXiv ID:** {paper['id']}")
            else:
                st.write("No relevant papers found.")
    
    elif mode == "Explain Concepts":
        st.header("🧠 Concept Explanation")
        
        concept = st.text_input("Enter a concept to explain (e.g., 'transformer', 'attention mechanism'):")
        
        if st.button("Explain") and concept:
            explanation = explain_concept(concept)
            st.write("**Explanation:**")
            st.write(explanation)
    
    elif mode == "Paper Summary":
        st.header("📄 Paper Summary")
        
        paper_options = [f"{row['title']} - {row['authors']}" for _, row in df.iterrows()]
        selected_paper = st.selectbox("Select a paper to summarize:", paper_options)
        
        if st.button("Summarize") and selected_paper:
            # Find the selected paper
            paper_index = paper_options.index(selected_paper)
            paper = df.iloc[paper_index]
            
            summary = summarize_paper(paper)
            st.write("**Summary:**")
            st.write(summary)
    
    elif mode == "Visualizations":
        st.header("📊 Research Paper Analytics")
        
        # Category distribution
        fig = create_category_visualization(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Papers table
        st.write("**All Papers:**")
        st.dataframe(df[['title', 'authors', 'categories']], use_container_width=True)
    
    # Follow-up questions
    st.sidebar.write("---")
    st.sidebar.write("**Quick Questions:**")
    quick_questions = [
        "What is deep learning?",
        "Explain neural networks",
        "What is machine learning?",
        "Tell me about AI ethics"
    ]
    
    for question in quick_questions:
        if st.sidebar.button(question, key=question):
            explanation = explain_concept(question.replace("What is ", "").replace("Tell me about ", "").replace("Explain ", ""))
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {explanation}")

if __name__ == "__main__":
    main()