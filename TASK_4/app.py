# Task 4: arXiv Expert Chatbot

import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict
import plotly.express as px
from collections import Counter

# Only using scikit-learn for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class UltraSimpleArXivChatbot:
    """Ultra simple chatbot with minimal dependencies"""
    
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.papers_text = []
        self.initialized = False
    
    def load_data(self, max_papers: int = 2000, category: str = 'cs') -> bool:
        """Load and filter ArXiv data"""
        try:
            st.info(f"Loading up to {max_papers} papers from category '{category}'...")
            papers = []
            
            with open(self.json_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    
                    if len(papers) >= max_papers:
                        break
                    
                    try:
                        paper = json.loads(line)
                        
                        # Filter by category
                        if 'categories' in paper and paper['categories']:
                            if paper['categories'].startswith(category):
                                papers.append(paper)
                    
                    except json.JSONDecodeError:
                        continue
                    
                    # Progress update every 10k lines
                    if line_count % 10000 == 0:
                        st.info(f"Processed {line_count} lines, found {len(papers)} relevant papers...")
            
            if papers:
                self.df = pd.DataFrame(papers)
                st.success(f"Successfully loaded {len(papers)} papers!")
                return True
            else:
                st.error(f"No papers found for category '{category}'. Try a different category.")
                return False
                
        except FileNotFoundError:
            st.error(f"❌ File not found: {self.json_path}")
            st.info("Please check the file path in the sidebar.")
            return False
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def preprocess_papers(self):
        """Preprocess papers for search"""
        if self.df is None or self.df.empty:
            return
        
        st.info("Preprocessing papers for search...")
        progress_bar = st.progress(0)
        self.papers_text = []
        
        for i, (_, row) in enumerate(self.df.iterrows()):
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            
            # Combine title and abstract
            text = f"{title} {abstract}"
            
            # Basic cleaning
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Remove non-alphabetic chars
            text = ' '.join(text.split())  # Remove extra whitespace
            text = text.lower()
            
            self.papers_text.append(text)
            
            # Update progress
            progress_bar.progress((i + 1) / len(self.df))
        
        # Create TF-IDF matrix
        st.info("Creating search index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.papers_text)
        st.success("✅ Preprocessing complete! Ready to search.")
    
    def search_papers(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for papers using TF-IDF similarity"""
        if self.tfidf_vectorizer is None:
            return []
