# Task 3: Medical Q&A Chatbot
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from groq import Groq

# Initialize Groq client
client = Groq(api_key="YOUR_GROQ_API_KEY")

# Medical entities for simple NER
MEDICAL_ENTITIES = {
    'symptoms': ['fever', 'headache', 'cough', 'pain', 'nausea', 'fatigue', 'dizziness'],
    'diseases': ['diabetes', 'hypertension', 'cancer', 'heart disease', 'flu', 'covid'],
    'treatments': ['medication', 'surgery', 'therapy', 'rest', 'exercise', 'diet']
}

def load_medquad_data():
    """Simulate loading MedQuAD data - replace with actual data loading"""
    # For demo purposes, using sample medical Q&A data
    sample_data = [
        {"question": "What is diabetes?", "answer": "Diabetes is a group of metabolic disorders characterized by high blood sugar levels."},
        {"question": "What are symptoms of flu?", "answer": "Common flu symptoms include fever, cough, body aches, headache, and fatigue."},
        {"question": "How to treat hypertension?", "answer": "Hypertension treatment includes lifestyle changes, medication, and regular monitoring."},
        {"question": "What causes headaches?", "answer": "Headaches can be caused by stress, dehydration, lack of sleep, or underlying conditions."},
        {"question": "How to prevent heart disease?", "answer": "Heart disease prevention includes healthy diet, regular exercise, and avoiding smoking."}
    ]
    return pd.DataFrame(sample_data)

def extract_medical_entities(text):
    """Simple medical entity recognition"""
    text_lower = text.lower()
    entities = {'symptoms': [], 'diseases': [], 'treatments': []}
    
    for category, terms in MEDICAL_ENTITIES.items():
        for term in terms:
            if term in text_lower:
                entities[category].append(term)
    
    return entities

def find_relevant_answers(query, df, top_k=3):
    """Find relevant answers using TF-IDF similarity"""
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Combine all questions and user query
    all_questions = df['question'].tolist() + [query]
    
    # Vectorize
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    
    # Calculate similarity
    query_vector = tfidf_matrix[-1]
    question_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    
    # Get top k similar questions
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Threshold for relevance
            results.append({
                'question': df.iloc[idx]['question'],
                'answer': df.iloc[idx]['answer'],
                'similarity': similarities[idx]
            })
    
    return results

def get_enhanced_response(query, relevant_answers):
    """Get enhanced response using Groq"""
    context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in relevant_answers])
    
    prompt = f"""You are a medical AI assistant. Based on the following medical knowledge and the user's question, provide a helpful response. 

Medical Knowledge:
{context}

User Question: {query}

Please provide a comprehensive answer based on the available information. If the question is outside your knowledge, suggest consulting a healthcare professional."""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("🏥 Medical Q&A Chatbot")
    st.write("Ask medical questions and get evidence-based answers")
    
    # Load data
    df = load_medquad_data()
    
    # User input
    query = st.text_input("Enter your medical question:")
    
    if st.button("Ask") and query:
        # Extract medical entities
        entities = extract_medical_entities(query)
        
        if any(entities.values()):
            st.write("**Detected Medical Terms:**")
            for category, terms in entities.items():
                if terms:
                    st.write(f"- {category.title()}: {', '.join(terms)}")
        
        # Find relevant answers
        relevant_answers = find_relevant_answers(query, df)
        
        if relevant_answers:
            st.write("**Most Relevant Information:**")
            for qa in relevant_answers[:2]:
                st.write(f"**Q:** {qa['question']}")
                st.write(f"**A:** {qa['answer']}")
                st.write(f"*Relevance: {qa['similarity']:.2f}*")
                st.write("---")
            
            # Get enhanced response
            enhanced_response = get_enhanced_response(query, relevant_answers)
            st.write("**Enhanced Response:**")
            st.write(enhanced_response)
        else:
            st.write("No relevant information found. Please consult a healthcare professional.")
    
    # Disclaimer
    st.write("---")
    st.warning("⚠️ This chatbot provides general information only. Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()