# Medical Q&A Chatbot using MedQuAD Dataset
# Complete implementation with retrieval, entity recognition, and Streamlit UI
# Without NLTK dependency - using built-in Python text processing

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from typing import List, Dict, Tuple
import requests
from datasets import load_dataset
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TextProcessor:
    """Custom text processing without NLTK dependencies"""
    
    def __init__(self):
        # Common English stopwords
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'two',
            'more', 'very', 'when', 'come', 'may', 'its', 'only', 'think', 'now',
            'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being',
            'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much',
            'your', 'way', 'well', 'down', 'could', 'good', 'also', 'until',
            'does', 'there', 'use', 'just', 'first', 'such', 'get', 'over',
            'think', 'back', 'other', 'take', 'than', 'only', 'little', 'state',
            'those', 'people', 'too', 'any', 'day', 'most', 'us', 'no', 'man',
            'year', 'work', 'life', 'right', 'high', 'might', 'came', 'show',
            'every', 'great', 'where', 'help', 'through', 'line', 'turn', 'cause',
            'same', 'mean', 'differ', 'move', 'right', 'boy', 'old', 'too',
            'does', 'tell', 'sentence', 'set', 'three', 'want', 'air', 'well',
            'also', 'play', 'small', 'end', 'put', 'home', 'read', 'hand',
            'port', 'large', 'spell', 'add', 'even', 'land', 'here', 'must',
            'big', 'high', 'such', 'follow', 'act', 'why', 'ask', 'men',
            'change', 'went', 'light', 'kind', 'off', 'need', 'house', 'picture',
            'try', 'again', 'animal', 'point', 'mother', 'world', 'near', 'build',
            'self', 'earth', 'father', 'head', 'stand', 'own', 'page', 'should',
            'country', 'found', 'answer', 'school', 'grow', 'study', 'still',
            'learn', 'plant', 'cover', 'food', 'sun', 'four', 'between', 'state',
            'keep', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree',
            'cross', 'farm', 'hard', 'start', 'might', 'story', 'saw', 'far',
            'sea', 'draw', 'left', 'late', 'run', 'dont', 'while', 'press',
            'close', 'night', 'real', 'life', 'few', 'north', 'open', 'seem',
            'together', 'next', 'white', 'children', 'begin', 'got', 'walk',
            'example', 'ease', 'paper', 'group', 'always', 'music', 'those',
            'both', 'mark', 'often', 'letter', 'until', 'mile', 'river', 'car',
            'feet', 'care', 'second', 'book', 'carry', 'took', 'science', 'eat',
            'room', 'friend', 'began', 'idea', 'fish', 'mountain', 'stop', 'once',
            'base', 'hear', 'horse', 'cut', 'sure', 'watch', 'color', 'face',
            'wood', 'main', 'enough', 'plain', 'girl', 'usual', 'young', 'ready',
            'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird',
            'soon', 'body', 'dog', 'family', 'direct', 'pose', 'leave', 'song',
            'measure', 'door', 'product', 'black', 'short', 'numeral', 'class',
            'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half',
            'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew',
            'pass', 'since', 'top', 'whole', 'king', 'space', 'heard', 'best',
            'hour', 'better', 'during', 'hundred', 'five', 'remember', 'step',
            'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast',
            'verb', 'sing', 'listen', 'six', 'table', 'travel', 'less', 'morning',
            'ten', 'simple', 'several', 'vowel', 'toward', 'war', 'lay', 'against',
            'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve',
            'appear', 'road', 'map', 'rain', 'rule', 'govern', 'pull', 'cold',
            'notice', 'voice', 'unit', 'power', 'town', 'fine', 'certain', 'fly',
            'fall', 'lead', 'cry', 'dark', 'machine', 'note', 'wait', 'plan',
            'figure', 'star', 'box', 'noun', 'field', 'rest', 'correct', 'able',
            'pound', 'done', 'beauty', 'drive', 'stood', 'contain', 'front',
            'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick', 'develop',
            'ocean', 'warm', 'free', 'minute', 'strong', 'special', 'mind',
            'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch',
            'multiply', 'nothing', 'course', 'stay', 'wheel', 'full', 'force',
            'blue', 'object', 'decide', 'surface', 'deep', 'moon', 'island',
            'foot', 'system', 'busy', 'test', 'record', 'boat', 'common', 'gold',
            'possible', 'plane', 'stead', 'dry', 'wonder', 'laugh', 'thousands',
            'ago', 'ran', 'check', 'game', 'shape', 'equate', 'hot', 'miss',
            'brought', 'heat', 'snow', 'tire', 'bring', 'yes', 'distant', 'fill',
            'east', 'paint', 'language', 'among'
        }
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace"""
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def simple_lemmatize(self, word: str) -> str:
        """Simple lemmatization by removing common suffixes"""
        # Common suffix patterns for medical terms
        suffixes = {
            'ies': 'y',
            'ied': 'y',
            'ies': 'y',
            'ing': '',
            'ed': '',
            's': '',
            'es': '',
        }
        
        word = word.lower()
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)] + replacement
        return word
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation but keep medical abbreviations
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Simple lemmatization
        tokens = [self.simple_lemmatize(token) for token in tokens if len(token) > 2]
        
        return ' '.join(tokens)

class MedicalEntityExtractor:
    """Extract medical entities from text using rule-based patterns"""
    
    def __init__(self):
        # Comprehensive medical entity patterns
        self.symptom_patterns = [
            # Pain types
            r'\b(?:pain|ache|aching|hurt|hurting|sore|soreness|tender|tenderness)\b',
            r'\b(?:chest pain|back pain|stomach pain|abdominal pain|joint pain|muscle pain|headache|migraine)\b',
            r'\b(?:neck pain|shoulder pain|knee pain|leg pain|arm pain|hip pain|foot pain)\b',
            
            # General symptoms
            r'\b(?:fever|temperature|chills|sweating|fatigue|tired|weakness|dizzy|dizziness|nausea|vomiting)\b',
            r'\b(?:cough|coughing|shortness of breath|breathless|wheezing|chest tightness)\b',
            r'\b(?:rash|itching|swelling|inflammation|bleeding|bruising|numbness|tingling)\b',
            r'\b(?:constipation|diarrhea|heartburn|indigestion|loss of appetite|weight loss|weight gain)\b',
            r'\b(?:insomnia|sleep problems|blurred vision|double vision|hearing loss|tinnitus)\b',
            
            # Specific symptoms
            r'\b(?:runny nose|stuffy nose|sneezing|sore throat|hoarse voice|difficulty swallowing)\b',
            r'\b(?:frequent urination|painful urination|blood in urine|incontinence)\b',
            r'\b(?:irregular heartbeat|palpitations|high blood pressure|low blood pressure)\b'
        ]
        
        self.disease_patterns = [
            # Common diseases
            r'\b(?:diabetes|cancer|tumor|hypertension|asthma|arthritis|pneumonia|bronchitis|influenza|flu)\b',
            r'\b(?:covid|coronavirus|cold|strep throat|sinusitis|allergies|infection|bacterial|viral)\b',
            
            # Chronic conditions
            r'\b(?:heart disease|kidney disease|liver disease|lung disease|thyroid disease)\b',
            r'\b(?:alzheimer|parkinson|multiple sclerosis|epilepsy|stroke|seizure|migraine)\b',
            r'\b(?:depression|anxiety|bipolar|schizophrenia|adhd|autism|dementia)\b',
            
            # Specific conditions
            r'\b(?:gastritis|ulcer|hernia|gallstones|appendicitis|colitis|crohn)\b',
            r'\b(?:osteoporosis|fibromyalgia|lupus|rheumatoid arthritis|gout)\b',
            r'\b(?:cataracts|glaucoma|macular degeneration|retinopathy)\b',
            
            # General terms
            r'\b(?:syndrome|disorder|condition|disease|illness|pathology|malignancy)\b'
        ]
        
        self.treatment_patterns = [
            # Medications
            r'\b(?:medication|medicine|drug|prescription|pill|tablet|capsule|injection|vaccine)\b',
            r'\b(?:antibiotics|painkillers|analgesics|anti-inflammatory|steroids|insulin|chemotherapy)\b',
            r'\b(?:antidepressants|antihistamines|beta blockers|ace inhibitors|diuretics)\b',
            
            # Procedures
            r'\b(?:surgery|operation|procedure|biopsy|endoscopy|colonoscopy|mammography)\b',
            r'\b(?:x-ray|ct scan|mri|ultrasound|ecg|ekg|blood test|urine test)\b',
            
            # Therapies
            r'\b(?:therapy|treatment|rehabilitation|physical therapy|occupational therapy)\b',
            r'\b(?:radiation|dialysis|transfusion|transplant|implant|pacemaker)\b',
            
            # Alternative treatments
            r'\b(?:acupuncture|chiropractic|massage|meditation|counseling|psychotherapy)\b'
        ]
        
        # Compile patterns for better performance
        self.symptom_regex = re.compile('|'.join(self.symptom_patterns), re.IGNORECASE)
        self.disease_regex = re.compile('|'.join(self.disease_patterns), re.IGNORECASE)
        self.treatment_regex = re.compile('|'.join(self.treatment_patterns), re.IGNORECASE)
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': []
        }
        
        # Extract symptoms
        symptom_matches = self.symptom_regex.findall(text)
        entities['symptoms'] = list(set([s.lower() for s in symptom_matches if len(s) > 2]))
        
        # Extract diseases
        disease_matches = self.disease_regex.findall(text)
        entities['diseases'] = list(set([d.lower() for d in disease_matches if len(d) > 2]))
        
        # Extract treatments
        treatment_matches = self.treatment_regex.findall(text)
        entities['treatments'] = list(set([t.lower() for t in treatment_matches if len(t) > 2]))
        
        return entities

class MedicalQARetriever:
    """Medical Q&A Retrieval system using TF-IDF and cosine similarity"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.qa_data = None
        self.text_processor = TextProcessor()
        self.entity_extractor = MedicalEntityExtractor()
        
    def load_dataset(self) -> pd.DataFrame:
        """Load MedQuAD dataset from Hugging Face"""
        try:
            with st.spinner("Loading MedQuAD dataset from Hugging Face..."):
                # Load dataset from Hugging Face
                dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
                df = pd.DataFrame(dataset)
                
                # Clean and prepare data
                df = df.dropna(subset=['question', 'answer'])
                df = df[df['question'].str.len() > 10]  # Filter very short questions
                df = df[df['answer'].str.len() > 20]    # Filter very short answers
                
                # Preprocess questions for better matching
                df['processed_question'] = df['question'].apply(self.text_processor.preprocess_text)
                
                # Truncate very long answers for display
                df['display_answer'] = df['answer'].apply(lambda x: x[:1000] if len(x) > 1000 else x)
                
                st.success(f"Successfully loaded {len(df)} medical Q&A pairs!")
                return df
                
        except Exception as e:
            st.warning(f"Could not load dataset from Hugging Face: {str(e)}")
            st.info("Using sample medical data as fallback...")
            return self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample medical Q&A data as fallback"""
        sample_data = {
            'question': [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?",
                "What causes chest pain?",
                "What are the side effects of aspirin?",
                "How to prevent heart disease?",
                "What is the treatment for pneumonia?",
                "What are the symptoms of COVID-19?",
                "How is asthma diagnosed?",
                "What causes headaches?",
                "How to treat fever?",
                "What are the symptoms of depression?",
                "How is high cholesterol treated?",
                "What causes back pain?",
                "What are the side effects of antibiotics?",
                "How to prevent stroke?",
                "What is the treatment for arthritis?",
                "What are the symptoms of allergies?",
                "How is kidney disease diagnosed?",
                "What causes stomach pain?",
                "How to treat insomnia?"
            ],
            'answer': [
                "Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, extreme fatigue, blurred vision, slow-healing cuts or bruises, and tingling or numbness in hands and feet. Type 1 diabetes symptoms often develop quickly, while type 2 diabetes symptoms develop gradually over time.",
                
                "Hypertension treatment typically involves lifestyle changes and medications. Lifestyle modifications include reducing salt intake, maintaining healthy weight, regular exercise, limiting alcohol, and stress management. Medications may include ACE inhibitors, ARBs, diuretics, beta-blockers, or calcium channel blockers, depending on individual patient needs.",
                
                "Chest pain can result from various conditions including heart problems (angina, heart attack, pericarditis), lung issues (pneumonia, pulmonary embolism, pleurisy), digestive problems (acid reflux, esophageal spasm), musculoskeletal issues (muscle strain, rib fractures), or anxiety and panic attacks. Immediate medical attention is needed for severe chest pain.",
                
                "Common side effects of aspirin include stomach irritation, increased bleeding risk, nausea, heartburn, and gastrointestinal upset. Serious side effects may include stomach ulcers, gastrointestinal bleeding, allergic reactions, ringing in ears (tinnitus), and in rare cases, Reye's syndrome in children. Always consult healthcare providers before starting aspirin therapy.",
                
                "Heart disease prevention includes maintaining a healthy diet low in saturated fats and trans fats, regular physical activity (150 minutes moderate exercise weekly), not smoking, limiting alcohol consumption, managing stress, controlling blood pressure and cholesterol levels, maintaining healthy weight, and managing diabetes if present.",
                
                "Pneumonia treatment depends on the cause and severity. Bacterial pneumonia is treated with antibiotics, while viral pneumonia requires supportive care. Treatment includes rest, increased fluid intake, fever reducers, and cough medicine. Severe cases may require hospitalization, oxygen therapy, or IV antibiotics. Pneumonia vaccines can help prevent certain types.",
                
                "COVID-19 symptoms include fever, cough, shortness of breath, fatigue, muscle aches, headache, loss of taste or smell, sore throat, congestion, nausea, vomiting, and diarrhea. Symptoms can range from mild to severe and may appear 2-14 days after exposure. Seek immediate medical attention for difficulty breathing or persistent chest pain.",
                
                "Asthma diagnosis involves medical history review, physical examination, and lung function tests. Tests include spirometry to measure airflow, peak flow measurement, fractional exhaled nitric oxide test, and sometimes chest X-rays or allergy tests. Doctors assess symptoms like wheezing, shortness of breath, chest tightness, and coughing patterns.",
                
                "Headaches can be caused by stress, tension, dehydration, lack of sleep, eye strain, sinus problems, hormonal changes, certain foods, medications, high blood pressure, or underlying medical conditions. Primary headaches include tension headaches, migraines, and cluster headaches. Secondary headaches result from other medical conditions.",
                
                "Fever treatment includes rest, increased fluid intake to prevent dehydration, fever-reducing medications like acetaminophen or ibuprofen (following dosage instructions), wearing lightweight clothing, and monitoring temperature. Seek medical care for very high fever (over 103°F/39.4°C), persistent fever, or concerning symptoms.",
                
                "Depression symptoms include persistent sadness, loss of interest in activities, changes in appetite and sleep, fatigue, difficulty concentrating, feelings of worthlessness or guilt, and thoughts of death or suicide. Physical symptoms may include headaches, digestive issues, and chronic pain. Professional help is important for proper diagnosis and treatment.",
                
                "High cholesterol treatment includes dietary changes (reducing saturated and trans fats, increasing fiber), regular exercise, weight management, and sometimes medications. Statins are commonly prescribed to lower LDL cholesterol. Other medications include bile acid sequestrants, cholesterol absorption inhibitors, and PCSK9 inhibitors for severe cases.",
                
                "Back pain causes include muscle strain, herniated discs, spinal stenosis, arthritis, osteoporosis, poor posture, obesity, stress, and sometimes underlying conditions like kidney stones or infections. Mechanical causes are most common, often resulting from lifting, sudden movements, or prolonged sitting or standing.",
                
                "Antibiotic side effects include nausea, diarrhea, stomach upset, yeast infections, allergic reactions, and antibiotic-associated colitis. Some antibiotics may cause sun sensitivity, dizziness, or interactions with other medications. Taking probiotics and completing the full course as prescribed can help minimize side effects.",
                
                "Stroke prevention includes controlling blood pressure, managing cholesterol, not smoking, limiting alcohol, maintaining healthy weight, regular exercise, managing diabetes, treating atrial fibrillation, taking prescribed medications like blood thinners when appropriate, and eating a diet rich in fruits and vegetables while limiting sodium.",
                
                "Arthritis treatment varies by type but may include medications (NSAIDs, disease-modifying drugs, biologics), physical therapy, occupational therapy, weight management, regular exercise, heat/cold therapy, joint protection techniques, and sometimes surgery. Treatment plans are individualized based on arthritis type and severity.",
                
                "Allergy symptoms include sneezing, runny or stuffy nose, itchy or watery eyes, skin rash or hives, swelling, digestive issues, and in severe cases, anaphylaxis. Symptoms depend on the allergen and exposure route. Common allergens include pollen, dust mites, pet dander, foods, and medications.",
                
                "Kidney disease diagnosis involves blood tests (creatinine, BUN, GFR), urine tests (protein, blood, glucose), imaging studies (ultrasound, CT scan), and sometimes kidney biopsy. Early detection is important as kidney disease often has no symptoms until advanced stages. Regular screening is important for high-risk individuals.",
                
                "Stomach pain causes include indigestion, gastritis, ulcers, food poisoning, gastroenteritis, appendicitis, gallstones, kidney stones, inflammatory bowel disease, or stress. Pain location, timing, and associated symptoms help determine the cause. Persistent or severe pain requires medical evaluation.",
                
                "Insomnia treatment includes good sleep hygiene (regular sleep schedule, comfortable environment, avoiding caffeine/screens before bed), relaxation techniques, cognitive behavioral therapy for insomnia (CBT-I), and sometimes short-term sleep medications. Addressing underlying causes like stress, anxiety, or medical conditions is important."
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df['processed_question'] = df['question'].apply(self.text_processor.preprocess_text)
        df['display_answer'] = df['answer']
        
        st.info(f"Using {len(df)} sample medical Q&A pairs")
        return df
    
    def build_index(self):
        """Build TF-IDF index for questions"""
        if self.qa_data is None:
            self.qa_data = self.load_dataset()
            
        # Create TF-IDF vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),  # Include trigrams for medical terms
            min_df=1,
            max_df=0.95,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Combine original and processed questions for better matching
        combined_questions = (self.qa_data['question'] + ' ' + self.qa_data['processed_question']).values
        
        # Fit TF-IDF on combined questions
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_questions)
        
        return len(self.qa_data)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant Q&A pairs"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
            
        # Preprocess query
        processed_query = self.text_processor.preprocess_text(query)
        combined_query = query + ' ' + processed_query
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([combined_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k most similar questions
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for better recall
                results.append({
                    'question': self.qa_data.iloc[idx]['question'],
                    'answer': self.qa_data.iloc[idx]['answer'],
                    'display_answer': self.qa_data.iloc[idx]['display_answer'],
                    'similarity': similarities[idx],
                    'index': idx
                })
        
        return results

class MedicalChatbot:
    """Main Medical Chatbot class"""
    
    def __init__(self):
        self.retriever = MedicalQARetriever()
        self.entity_extractor = MedicalEntityExtractor()
        
    def initialize(self):
        """Initialize the chatbot"""
        dataset_size = self.retriever.build_index()
        return dataset_size
        
    def get_response(self, user_question: str) -> Dict:
        """Get response for user question"""
        # Extract entities from question
        entities = self.entity_extractor.extract_entities(user_question)
        
        # Search for relevant Q&A pairs
        results = self.retriever.search(user_question, top_k=5)
        
        response = {
            'entities': entities,
            'results': results,
            'has_results': len(results) > 0,
            'query_processed': self.retriever.text_processor.preprocess_text(user_question)
        }
        
        return response

# Streamlit UI
def main():
    st.set_page_config(
        page_title="MedQuAD Medical Q&A Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with modern medical theme
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .result-box {
        background: white;
        padding: 1.5rem;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .result-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .similarity-score {
        background: linear-gradient(135deg, #2E86AB, #4A90A4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .entity-tag {
        background-color: #e8f4fd;
        color: #2E86AB;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
        border: 1px solid #2E86AB;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #f6d55c;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 2rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🏥 MedQuAD Medical Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask medical questions and get evidence-based answers from 47,457+ medical Q&A pairs</p>', unsafe_allow_html=True)
    
    # Medical disclaimer banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 1rem; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <strong>⚠️ Medical Disclaimer:</strong> This is for educational purposes only. Always consult healthcare professionals for medical advice.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalChatbot()
        with st.spinner("🔄 Loading MedQuAD medical database..."):
            dataset_size = st.session_state.chatbot.initialize()
            st.session_state.dataset_size = dataset_size
    
    # Sidebar with comprehensive information
    with st.sidebar:
        st.markdown("### 📊 System Information")
        
        # Dataset statistics
        if 'dataset_size' in st.session_state:
            st.markdown(f"""
            <div class="stats-card">
                <h4 style="color: #2E86AB; margin: 0;">📚 Dataset Stats</h4>
                <p style="margin: 0.5rem 0;"><strong>{st.session_state.dataset_size:,}</strong> Q&A Pairs</p>
                <p style="margin: 0.5rem 0;"><strong>12</strong> NIH Sources</p>
                <p style="margin: 0.5rem 0;"><strong>37</strong> Question Types</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🏷️ Entity Recognition")
        st.markdown("""
        The system automatically identifies:
        
        **🔸 Symptoms**
        - Pain, fever, fatigue, nausea
        - Respiratory symptoms
        - Neurological symptoms
        
        **🔸 Diseases** 
        - Chronic conditions
        - Infections
        - Mental health conditions
        
        **🔸 Treatments**
        - Medications
        - Procedures
        - Therapies
        """)
        
        st.markdown("---")
        
        st.markdown("### 🔍 Search Features")
        st.markdown("""
        - **TF-IDF Vectorization**: Advanced text matching
        - **Cosine Similarity**: Relevance scoring
        - **N-gram Analysis**: Context understanding
        - **Multi-answer Results**: Multiple perspectives
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Usage Tips")
        st.markdown("""
        - Be specific in your questions
        - Include relevant symptoms
        - Use medical terminology when known
        - Ask about symptoms, treatments, or prevention
        """)
        
        st.markdown("---")
        
        st.markdown("### 📚 Data Sources")
        st.markdown("""
        - Cancer.gov
        - NIDDK.nih.gov
        - MedlinePlus
        - GARD (NIH)
        - And 8+ more NIH sites
        """)
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Ask Your Medical Question")
        
        # Example questions with categories
        example_categories = {
            "General Symptoms": [
                "What are the symptoms of diabetes?",
                "What causes chest pain?",
                "How to treat fever?"
            ],
            "Treatment & Medications": [
                "How is high blood pressure treated?",
                "What are the side effects of aspirin?",
                "What is the treatment for pneumonia?"
            ],
            "Prevention & Care": [
                "How to prevent heart disease?",
                "How to prevent stroke?",
                "What foods help lower cholesterol?"
            ],
            "Diagnosis & Tests": [
                "How is asthma diagnosed?",
                "What tests are used for diabetes?",
                "How is kidney disease detected?"
            ]
        }
        
        # Category selection
        selected_category = st.selectbox(
            "Choose a question category:",
            ["Select a category..."] + list(example_categories.keys())
        )
        
        # Example question selection
        if selected_category != "Select a category...":
            selected_example = st.selectbox(
                f"Example {selected_category.lower()} questions:",
                [""] + example_categories[selected_category]
            )
        else:
            selected_example = ""
        
        # Question input
        if selected_example:
            user_question = st.text_area(
                "Your medical question:",
                value=selected_example,
                height=100,
                help="Edit the example question or write your own"
            )
        else:
            user_question = st.text_area(
                "Your medical question:",
                height=100,
                placeholder="E.g., What are the symptoms of pneumonia? How is diabetes treated?",
                help="Ask about symptoms, treatments, causes, prevention, or diagnosis"
            )
        
        # Search controls
        col_search, col_clear = st.columns([3, 1])
        
        with col_search:
            search_button = st.button("🔍 Search Medical Database", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pop('last_response', None)
                st.session_state.pop('last_question', None)
                st.rerun()
        
        # Process search
        if search_button:
            if user_question.strip():
                with st.spinner("🔍 Searching through medical database..."):
                    response = st.session_state.chatbot.get_response(user_question)
                    
                    # Store response in session state
                    st.session_state.last_response = response
                    st.session_state.last_question = user_question
                    
                    # Show success message
                    if response['has_results']:
                        st.success(f"✅ Found {len(response['results'])} relevant medical answers!")
                    else:
                        st.warning("⚠️ No highly relevant results found. Try rephrasing your question.")
                    
                    st.rerun()
            else:
                st.error("❌ Please enter a medical question.")
    
    with col2:
        st.markdown("### 📈 Search Analytics")
        
        if 'last_response' in st.session_state:
            response = st.session_state.last_response
            
            # Results metrics
            st.markdown(f"""
            <div class="stats-card">
                <h4 style="color: #2E86AB; margin: 0;">🎯 Results</h4>
                <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold; color: #2E86AB;">
                    {len(response['results'])}
                </p>
                <p style="margin: 0; color: #666;">Answers Found</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Entity metrics
            total_entities = sum(len(entities) for entities in response['entities'].values())
            st.markdown(f"""
            <div class="stats-card" style="margin-top: 1rem;">
                <h4 style="color: #2E86AB; margin: 0;">🏷️ Entities</h4>
                <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold; color: #2E86AB;">
                    {total_entities}
                </p>
                <p style="margin: 0; color: #666;">Terms Detected</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Similarity score
            if response['results']:
                max_similarity = max(r['similarity'] for r in response['results'])
                st.markdown(f"""
                <div class="stats-card" style="margin-top: 1rem;">
                    <h4 style="color: #2E86AB; margin: 0;">📊 Best Match</h4>
                    <p style="margin: 0.5rem 0; font-size: 1.5rem; font-weight: bold; color: #2E86AB;">
                        {max_similarity:.1%}
                    </p>
                    <p style="margin: 0; color: #666;">Similarity Score</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.markdown("""
            <div class="stats-card">
                <h4 style="color: #2E86AB; margin: 0;">💡 Ready to Help</h4>
                <p style="margin: 0.5rem 0; color: #666;">
                    Enter a medical question to see analytics
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display results
    if 'last_response' in st.session_state and 'last_question' in st.session_state:
        response = st.session_state.last_response
        
        st.markdown("---")
        
        # Query processing info
        with st.expander("🔍 Query Processing Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Question:**")
                st.code(st.session_state.last_question)
            with col2:
                st.write("**Processed Query:**")
                st.code(response.get('query_processed', 'N/A'))
        
        # Display extracted entities
        if any(response['entities'].values()):
            st.markdown("### 🏷️ Detected Medical Entities")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if response['entities']['symptoms']:
                    st.markdown("**🔸 Symptoms:**")
                    for symptom in response['entities']['symptoms']:
                        st.markdown(f'<span class="entity-tag">{symptom.title()}</span>', 
                                  unsafe_allow_html=True)
            
            with col2:
                if response['entities']['diseases']:
                    st.markdown("**🔸 Diseases:**")
                    for disease in response['entities']['diseases']:
                        st.markdown(f'<span class="entity-tag">{disease.title()}</span>', 
                                  unsafe_allow_html=True)
            
            with col3:
                if response['entities']['treatments']:
                    st.markdown("**🔸 Treatments:**")
                    for treatment in response['entities']['treatments']:
                        st.markdown(f'<span class="entity-tag">{treatment.title()}</span>', 
                                  unsafe_allow_html=True)
            
            st.markdown("")
        
        # Display results
        if response['has_results']:
            st.markdown("### 📋 Medical Information Results")
            
            for i, result in enumerate(response['results'], 1):
                with st.container():
                    # Result header with similarity score
                    col_title, col_score = st.columns([4, 1])
                    
                    with col_title:
                        st.markdown(f"""
                        <div class="result-box">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h4 style="margin: 0; color: #2E86AB;">📌 Result {i}</h4>
                                <span class="similarity-score">Match: {result['similarity']:.1%}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Question and answer
                    st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <h5 style="color: #1f77b4; margin-bottom: 0.5rem; font-size: 1.1rem;">
                                ❓ {result['question']}
                            </h5>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display answer (truncated or full)
                    display_text = result['display_answer']
                    full_text = result['answer']
                    
                    if len(full_text) > 800:
                        st.markdown(f"""
                        <div style="line-height: 1.6; margin-bottom: 1rem;">
                            💡 <strong>Answer:</strong><br>
                            {display_text}...
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Expandable full answer
                        with st.expander(f"📖 Read Full Answer for Result {i}"):
                            st.markdown(f"""
                            <div style="line-height: 1.6; padding: 1rem; background: #f8f9fa; border-radius: 5px;">
                                {full_text}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="line-height: 1.6; margin-bottom: 1rem;">
                            💡 <strong>Answer:</strong><br>
                            {full_text}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("")
        
        else:
            # No results found
            st.markdown("### ❌ No Results Found")
            st.markdown("""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 1.5rem; border-radius: 10px; text-align: center;">
                <h4>🔍 No highly relevant results found</h4>
                <p><strong>Suggestions to improve your search:</strong></p>
                <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                    <li>Try using more specific medical terms</li>
                    <li>Include specific symptoms or conditions</li>
                    <li>Rephrase your question differently</li>
                    <li>Ask about symptoms, treatments, causes, or prevention</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Medical disclaimer
        st.markdown("""
        <div class="warning-box">
            <div style="text-align: center;">
                <h4 style="margin-top: 0; color: #856404;">⚠️ Important Medical Disclaimer</h4>
                <p style="margin-bottom: 0;">
                    <strong>This information is for educational purposes only and should not replace professional medical advice.</strong><br>
                    Always consult with qualified healthcare providers for medical diagnosis, treatment decisions, and personalized medical advice.
                    In case of medical emergencies, contact emergency services immediately.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Session management footer
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"**🕒 Session:** {current_time}")
        
    with col2:
        if 'last_response' in st.session_state:
            if st.button("💾 Export Results"):
                # Create export data
                export_data = {
                    'question': st.session_state.last_question,
                    'timestamp': current_time,
                    'results_count': len(st.session_state.last_response['results']),
                    'entities': st.session_state.last_response['entities']
                }
                st.json(export_data)
        
    with col3:
        if st.button("🔄 New Session"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key.startswith('last_'):
                    del st.session_state[key]
            st.success("✅ Session reset!")
            st.rerun()
    
    # Footer with credits and information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem;">
        <h4 style="color: #2E86AB;">🏥 MedQuAD Medical Q&A Chatbot</h4>
        <p>
            Built with ❤️ using <strong>Streamlit</strong> | 
            Data from <strong>MedQuAD Dataset</strong> (NIH) | 
            <strong>47,457+</strong> Medical Q&A Pairs
        </p>
        <p>
            <a href="https://github.com/abachaa/MedQuAD" target="_blank" style="color: #2E86AB;">
                📚 Dataset Source
            </a> | 
            <a href="https://pubmed.ncbi.nlm.nih.gov/30649181/" target="_blank" style="color: #2E86AB;">
                📄 Research Paper
            </a>
        </p>
        <p style="font-size: 0.8rem; color: #999;">
            For educational and research purposes only. Not a substitute for professional medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()