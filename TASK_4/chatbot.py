# Ultra Simple ArXiv Chatbot - No External NLP Dependencies
# Perfect for quick testing and getting started

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
        
        # Preprocess query
        query = re.sub(r'[^a-zA-Z\s]', ' ', query)
        query = ' '.join(query.split()).lower()
        
        if not query.strip():
            return []
        
        try:
            # Get query vector
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    paper = self.df.iloc[idx].to_dict()
                    paper['similarity_score'] = similarities[idx]
                    results.append(paper)
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def simple_summarize(self, text: str, max_chars: int = 300) -> str:
        """Simple text summarization without NLTK"""
        if not text or len(text) < 100:
            return text
        
        # Split by periods for sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        # Take first sentence and try to add more until we hit the limit
        summary = sentences[0]
        
        for sentence in sentences[1:]:
            if len(summary + '. ' + sentence) <= max_chars:
                summary += '. ' + sentence
            else:
                break
        
        return summary + '.' if not summary.endswith('.') else summary
    
    def answer_query(self, query: str) -> str:
        """Generate answer based on top papers"""
        papers = self.search_papers(query, top_k=3)
        
        if not papers:
            return """I couldn't find relevant papers to answer your question. 
            
Try:
• Using different keywords
• Being more specific
• Checking if papers in this category cover your topic"""
        
        answer = f"📚 Found {len(papers)} relevant papers:\n\n"
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Unknown Title')
            abstract = paper.get('abstract', '')
            score = paper.get('similarity_score', 0)
            
            # Summarize abstract
            if abstract:
                summary = self.simple_summarize(abstract, 200)
            else:
                summary = "No abstract available."
            
            answer += f"**{i}. {title}**\n"
            answer += f"*Relevance Score: {score:.3f}*\n\n"
            answer += f"{summary}\n\n"
            answer += "---\n\n"
        
        return answer
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions based on loaded papers"""
        if not self.initialized or self.df is None:
            return [
                "What is machine learning?",
                "How do neural networks work?",
                "What are transformers in AI?"
            ]
        
        # Extract common terms from titles for suggestions
        all_titles = ' '.join(self.df['title'].fillna('').astype(str))
        common_words = ['learning', 'network', 'algorithm', 'model', 'system', 'analysis']
        
        suggestions = []
        for word in common_words:
            if word in all_titles.lower():
                suggestions.append(f"What is {word}?")
                if len(suggestions) >= 6:
                    break
        
        return suggestions[:6]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None or self.df.empty:
            return {}
        
        stats = {
            'total_papers': len(self.df),
            'date_range': 'N/A',
            'top_categories': [],
            'avg_abstract_length': 0,
            'year_distribution': {}
        }
        
        # Date analysis
        if 'update_date' in self.df.columns:
            dates = pd.to_datetime(self.df['update_date'], errors='coerce')
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                years = valid_dates.dt.year
                stats['date_range'] = f"{years.min()} - {years.max()}"
                stats['year_distribution'] = years.value_counts().head(10).to_dict()
        
        # Category analysis
        if 'categories' in self.df.columns:
            categories = []
            for cat_str in self.df['categories'].fillna(''):
                if cat_str:
                    # Get first category
                    main_cat = cat_str.split()[0] if ' ' in cat_str else cat_str
                    if '.' in main_cat:
                        main_cat = main_cat.split('.')[0]
                    categories.append(main_cat)
            
            if categories:
                category_counts = Counter(categories)
                stats['top_categories'] = category_counts.most_common(10)
        
        # Abstract length analysis
        if 'abstract' in self.df.columns:
            abstract_lengths = [len(str(abstract)) for abstract in self.df['abstract'].fillna('')]
            if abstract_lengths:
                stats['avg_abstract_length'] = int(np.mean(abstract_lengths))
        
        return stats

def main():
    st.set_page_config(
        page_title="Ultra Simple ArXiv Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Ultra Simple ArXiv Chatbot")
    st.markdown("*A lightweight chatbot for exploring ArXiv papers - No complex dependencies!*")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    json_path = st.sidebar.text_input(
        "📁 ArXiv JSON File Path:",
        value=r"C:\Users\ASUS\Downloads\archive (13)\arxiv-metadata-oai-snapshot.json",
        help="Full path to your ArXiv dataset JSON file"
    )
    
    max_papers = st.sidebar.slider(
        "📊 Max Papers to Load:",
        min_value=500,
        max_value=5000,
        value=1500,
        step=250,
        help="Start small for testing, increase for better coverage"
    )
    
    category = st.sidebar.selectbox(
        "🏷️ Paper Category:",
        options=['cs', 'math', 'physics', 'stat', 'q-bio', 'econ', 'eess'],
        index=0,
        help="cs = Computer Science (recommended for testing)"
    )
    
    st.sidebar.markdown("---")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = UltraSimpleArXivChatbot(json_path)
    
    # Initialize button
    if st.sidebar.button("🚀 Initialize Chatbot", type="primary"):
        with st.spinner("Initializing chatbot..."):
            success = st.session_state.chatbot.load_data(max_papers, category)
            if success:
                st.session_state.chatbot.preprocess_papers()
                st.session_state.chatbot.initialized = True
                st.sidebar.success("✅ Chatbot ready!")
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to initialize!")
    
    # Status indicator
    if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot.initialized:
        st.sidebar.success("🟢 Chatbot is ready!")
        st.sidebar.info(f"📚 Loaded {len(st.session_state.chatbot.df)} papers")
    else:
        st.sidebar.warning("🟡 Chatbot not initialized")
    
    # Main content
    if not hasattr(st.session_state, 'chatbot') or not st.session_state.chatbot.initialized:
        # Welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("👆 **Please initialize the chatbot using the sidebar first.**")
            
            st.markdown("""
            ### 🚀 Getting Started:
            1. **Check your file path** - Make sure the ArXiv JSON file path is correct
            2. **Start small** - Begin with 1500 papers for testing
            3. **Choose Computer Science** - 'cs' category works well for demos
            4. **Click Initialize** - Wait for the processing to complete
            
            ### 🎯 What this chatbot can do:
            - 🔍 **Search papers** based on your questions
            - 📝 **Summarize research** in simple terms  
            - 💬 **Answer questions** about scientific concepts
            - 📊 **Show statistics** about the dataset
            """)
        
        with col2:
            st.markdown("### 📋 Sample Questions")
            sample_questions = [
                "What is machine learning?",
                "How do neural networks work?", 
                "What are transformers?",
                "Explain deep learning",
                "What is computer vision?",
                "How does NLP work?"
            ]
            
            for q in sample_questions:
                st.markdown(f"• *{q}*")
    
    else:
        # Main interface - Tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search Papers", "📊 Dataset Stats"])
        
        with tab1:
            st.header("💬 Chat with the Bot")
            
            # Sample questions
            sample_questions = st.session_state.chatbot.get_sample_questions()
            
            col1, col2 = st.columns([3, 1])
            with col2:
                st.markdown("**💡 Try asking:**")
                for q in sample_questions[:4]:
                    if st.button(q, key=f"sample_{q}"):
                        st.session_state['sample_question'] = q
            
            # Chat history
            if 'messages' not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Hello! 👋 I'm ready to help you explore scientific papers. What would you like to know?"}
                ]
            
            with col1:
                # Display messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Handle sample question clicks
                if 'sample_question' in st.session_state:
                    prompt = st.session_state['sample_question']
                    del st.session_state['sample_question']
                    
                    # Add to messages and process
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Searching papers and generating answer..."):
                            response = st.session_state.chatbot.answer_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    st.rerun()
                
                # Chat input
                if prompt := st.chat_input("Ask me about papers or scientific concepts..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate response
                    with st.chat_message("assistant"):
                        with st.spinner("Searching papers and generating answer..."):
                            response = st.session_state.chatbot.answer_query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with tab2:
            st.header("🔍 Paper Search")
            
            search_query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., machine learning, neural networks, computer vision",
                help="Use keywords related to your research interest"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                num_results = st.slider("Number of results:", 1, 10, 5)
            with col2:
                if st.button("🔍 Search", type="primary"):
                    st.session_state['perform_search'] = True
            
            if search_query and (st.session_state.get('perform_search', False) or 'last_search' not in st.session_state or st.session_state.get('last_search') != search_query):
                st.session_state['perform_search'] = False
                st.session_state['last_search'] = search_query
                
                with st.spinner("Searching papers..."):
                    papers = st.session_state.chatbot.search_papers(search_query, top_k=num_results)
                
                if papers:
                    st.success(f"✅ Found {len(papers)} relevant papers:")
                    st.markdown("---")
                    
                    for i, paper in enumerate(papers):
                        score = paper.get('similarity_score', 0)
                        title = paper.get('title', 'No Title')
                        
                        # Color code by relevance
                        if score > 0.3:
                            relevance_color = "🟢"
                        elif score > 0.15:
                            relevance_color = "🟡"
                        else:
                            relevance_color = "🔴"
                        
                        with st.expander(f"{relevance_color} {i+1}. {title}", expanded=(i==0)):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Relevance Score:** {score:.3f}")
                                st.markdown(f"**Authors:** {paper.get('authors', 'N/A')}")
                                
                                abstract = paper.get('abstract', 'No abstract available')
                                st.markdown("**Abstract:**")
                                if len(abstract) > 500:
                                    st.markdown(abstract[:500] + "...")
                                    if st.button(f"Show full abstract", key=f"full_abstract_{i}"):
                                        st.markdown(abstract)
                                else:
                                    st.markdown(abstract)
                            
                            with col2:
                                st.markdown(f"**Published:** {paper.get('update_date', 'N/A')}")
                                st.markdown(f"**Categories:** {paper.get('categories', 'N/A')}")
                                
                                # Generate summary
                                if abstract and len(abstract) > 100:
                                    with st.spinner("Generating summary..."):
                                        summary = st.session_state.chatbot.simple_summarize(abstract)
                                    st.markdown("**Quick Summary:**")
                                    st.info(summary)
                else:
                    st.warning("❌ No relevant papers found. Try:")
                    st.markdown("• Different keywords\n• More general terms\n• Check spelling")
        
        with tab3:
            st.header("📊 Dataset Statistics")
            
            stats = st.session_state.chatbot.get_statistics()
            
            if stats:
                # Overview metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📚 Total Papers", f"{stats['total_papers']:,}")
                with col2:
                    st.metric("📅 Date Range", stats['date_range'])
                with col3:
                    st.metric("📄 Avg Abstract Length", f"{stats['avg_abstract_length']:,} chars")
                
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    if stats['top_categories']:
                        st.subheader("🏷️ Top Categories")
                        categories, counts = zip(*stats['top_categories'][:8])
                        
                        fig = px.bar(
                            x=list(categories),
                            y=list(counts),
                            title="Distribution of Paper Categories",
                            labels={'x': 'Category', 'y': 'Number of Papers'},
                            color=list(counts),
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if stats['year_distribution']:
                        st.subheader("📈 Papers by Year")
                        years = list(stats['year_distribution'].keys())
                        counts = list(stats['year_distribution'].values())
                        
                        fig = px.line(
                            x=years,
                            y=counts,
                            title="Publication Trends",
                            labels={'x': 'Year', 'y': 'Number of Papers'},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sample papers table
                st.subheader("📄 Sample Papers")
                display_cols = ['title', 'authors', 'update_date', 'categories']
                available_cols = [col for col in display_cols if col in st.session_state.chatbot.df.columns]
                
                if available_cols:
                    sample_df = st.session_state.chatbot.df[available_cols].head(10)
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    
    # Footer with tips
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Pro Tips:**")
    st.sidebar.markdown("""
    • Start with 1500 papers for quick testing
    • Use specific scientific terms for better results  
    • Computer Science (cs) category has good variety
    • Try questions like "What is [concept]?"
    • Clear browser cache if issues persist
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built with Streamlit* 🚀")

if __name__ == "__main__":
    main()