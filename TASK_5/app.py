import os
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

# =====================
# Page configuration
# =====================
st.set_page_config(
    page_title="AI-Powered Customer Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# =====================
# Utilities
# =====================

def init_session_state():
    """Initialize all session_state keys safely."""
    defaults = {
        "conversation_history": [],
        "sentiment_history": [],
        "ai_agent": None,
        "example_input": "",
        "api_choice": "Fallback Mode",
        "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =====================
# Sentiment & Emotion Analyzer
# =====================
class EmotionAwareSentimentAnalyzer:
    def __init__(self):
        self.emotion_keywords = {
            'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid', 'hate'],
            'sad': ['sad', 'depressed', 'unhappy', 'disappointed', 'upset', 'heartbroken', 'miserable', 'terrible'],
            'happy': ['happy', 'excited', 'thrilled', 'delighted', 'pleased', 'satisfied', 'great', 'awesome', 'love'],
            'fearful': ['worried', 'scared', 'afraid', 'anxious', 'concerned', 'nervous', 'panic', 'terrified'],
            'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'],
            'confused': ['confused', 'unclear', 'lost', 'puzzled', 'perplexed', "don't understand", 'help'],
        }
        self.intensity_modifiers = {
            'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'totally': 1.6,
            'quite': 1.2, 'somewhat': 0.8, 'slightly': 0.6, 'a bit': 0.7,
            'absolutely': 1.7, 'completely': 1.6, 'incredibly': 1.5,
        }

    def analyze_sentiment(self, text: str) -> Dict:
        if not text or not text.strip():
            return {
                'sentiment': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0,
                'confidence': 0.0, 'emotions': {}, 'intensity': 'low', 'urgency_level': 'low',
            }

        t = text.lower()
        blob = TextBlob(t)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        emotions = self._detect_emotions(t)
        intensity = self._calculate_intensity(t, abs(polarity))
        urgency_level = self._calculate_urgency(t, sentiment, intensity, emotions)
        confidence = min(subjectivity * abs(polarity) * 2.0, 1.0)

        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': confidence,
            'emotions': emotions,
            'intensity': intensity,
            'urgency_level': urgency_level,
        }

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        emotions: Dict[str, float] = {}
        words = text.split()
        for emotion, keywords in self.emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
                    if keyword in words:
                        idx = words.index(keyword)
                        if idx > 0:
                            prev = words[idx - 1]
                            if prev in self.intensity_modifiers:
                                score *= self.intensity_modifiers[prev]
            if score > 0:
                emotions[emotion] = min(score / 3.0, 1.0)
        return emotions

    def _calculate_intensity(self, text: str, base_polarity: float) -> str:
        intensity_score = base_polarity
        caps_ratio = (sum(1 for c in text if c.isupper()) / len(text)) if text else 0
        if caps_ratio > 0.3:
            intensity_score *= 1.5
        exclam = text.count('!')
        if exclam > 0:
            intensity_score *= (1 + exclam * 0.2)
        if '!!!' in text or '???' in text:
            intensity_score *= 1.3
        if any(w in text for w in ['urgent', 'emergency', 'immediately', 'asap', 'critical']):
            intensity_score *= 1.4
        if intensity_score > 0.7:
            return 'high'
        elif intensity_score > 0.3:
            return 'medium'
        return 'low'

    def _calculate_urgency(self, text: str, sentiment: str, intensity: str, emotions: Dict[str, float]) -> str:
        urgency_score = 0
        high = ['emergency', 'urgent', 'critical', 'broken', 'not working', 'error', 'bug',
                'refund', 'cancel', 'billing', 'charged', 'payment', 'security', 'hack']
        medium = ['problem', 'issue', 'help', 'support', 'question', 'confused', 'stuck']
        for k in high:
            if k in text:
                urgency_score += 2
        for k in medium:
            if k in text:
                urgency_score += 1
        if sentiment == 'negative' and intensity == 'high':
            urgency_score += 3
        elif sentiment == 'negative' and intensity == 'medium':
            urgency_score += 2
        if emotions.get('angry', 0) > 0.7:
            urgency_score += 2
        if emotions.get('fearful', 0) > 0.6:
            urgency_score += 1
        if urgency_score >= 5:
            return 'critical'
        elif urgency_score >= 3:
            return 'high'
        elif urgency_score >= 1:
            return 'medium'
        return 'low'

# =====================
# AI Response Generator
# =====================
class AICustomerSupportAgent:
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-1.5-flash'):
        self.api_key = api_key
        self.model = None
        self.model_name = model_name
        try:
            if api_key:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
        except Exception:
            self.model = None

        self.fallback_responses = {
            'positive': {
                'high': "I'm thrilled to hear you're so happy! How can I help make your experience even better?",
                'medium': "I'm glad to hear you're pleased! How can I assist you further?",
                'low': "Thank you for your feedback! How can I help you today?",
            },
            'negative': {
                'critical': "I sincerely apologize for this critical issue. I'm escalating this immediately to our senior support team for urgent resolution.",
                'high': "I completely understand your frustration, and I'm truly sorry. Let me personally ensure we resolve this for you immediately.",
                'medium': "I'm sorry to hear you're having difficulties. Let me work with you to find a solution.",
                'low': "I apologize for the trouble. What specific issue can I assist you with?",
            },
            'neutral': {
                'high': "Thank you for reaching out! I'm here to help with whatever you need.",
                'medium': "Hello! What can I help you with today?",
                'low': "How can I assist you?",
            },
        }

    def generate_customer_support_response(self, user_message: str, sentiment_analysis: Dict) -> str:
        if not self.model:
            return self._get_fallback_response(sentiment_analysis)
        try:
            prompt = self._create_support_prompt(user_message, sentiment_analysis)
            response = self.model.generate_content(prompt)
            text = getattr(response, 'text', None)
            return text.strip() if text else self._get_fallback_response(sentiment_analysis)
        except Exception:
            return self._get_fallback_response(sentiment_analysis)

    def _create_support_prompt(self, user_message: str, sa: Dict) -> str:
        sentiment = sa['sentiment']
        intensity = sa['intensity']
        emotions = sa['emotions']
        urgency = sa['urgency_level']
        emotion_context = ""
        if emotions:
            lst = [f"{k} ({v:.2f})" for k, v in emotions.items() if v > 0.3]
            if lst:
                emotion_context = f"Detected emotions: {', '.join(lst)}. "
        return f"""
You are an expert customer support agent. Respond to the following customer message:

CUSTOMER MESSAGE: "{user_message}"

SENTIMENT ANALYSIS:
- Sentiment: {sentiment} ({intensity} intensity)
- Urgency: {urgency}
- {emotion_context}
- Confidence: {sa['confidence']:.2f}

INSTRUCTIONS:
- Be empathetic, professional, concise.
- Address issue directly and provide solutions.
- Match tone to emotional state and urgency.
"""

    def _get_fallback_response(self, sa: Dict) -> str:
        sentiment = sa['sentiment']
        urgency = sa['urgency_level']
        key = urgency if urgency in ['critical'] else sa['intensity']
        bucket = self.fallback_responses.get(sentiment, {})
        return bucket.get(key, "Thank you for contacting us. I'm here to help you.")

ANALYZER = EmotionAwareSentimentAnalyzer()

# =====================
# Sidebar: API config & controls
# =====================
with st.sidebar:
    st.header("🔑 API Configuration")
    st.session_state.api_choice = st.selectbox("Select AI Provider:", ["Gemini AI", "Fallback Mode"], index=0)
    resolved_api_key = (
        st.text_input("Enter Gemini API Key:", type="password")
        if st.session_state.api_choice == "Gemini AI" else ""
    )

    if st.session_state.api_choice == "Gemini AI":
        if not resolved_api_key:
            resolved_api_key = os.getenv("GOOGLE_API_KEY", "")
        if resolved_api_key and (not st.session_state.ai_agent or st.session_state.ai_agent.api_key != resolved_api_key):
            st.session_state.ai_agent = AICustomerSupportAgent(api_key=resolved_api_key)
            st.success("✅ Gemini connected")
        elif not resolved_api_key:
            st.info("ℹ️ Provide an API key to enable AI responses")
    else:
        st.session_state.ai_agent = AICustomerSupportAgent(api_key=None)
        st.info("🔄 Using fallback response mode")

    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.sentiment_history = []
        st.rerun()

# =====================
# Main layout
# =====================
st.title("🤖 AI-Powered Customer Support Chatbot")
st.markdown("*Advanced sentiment analysis with Gemini AI-powered responses*")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")

    chat_container = st.container()
    with chat_container:
        for user_msg, bot_msg, sa in st.session_state.conversation_history:
            st.write(f"**You:** {user_msg}")
            st.caption(f"Sentiment: {sa['sentiment']} | Intensity: {sa['intensity']} | Urgency: {sa['urgency_level']}")
            st.write(f"**Bot:** {bot_msg}")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Type your message:",
            value=st.session_state.example_input,
            placeholder="Describe your issue...",
            height=100,
        )
        st.session_state.example_input = ""
        submitted = st.form_submit_button("Send Message")

        if submitted and user_input.strip():
            with st.spinner("🤖 Analyzing and responding..."):
                sa = ANALYZER.analyze_sentiment(user_input)
                if st.session_state.ai_agent:
                    bot = st.session_state.ai_agent.generate_customer_support_response(user_input, sa)
                else:
                    bot = "I'm here to help! Please configure an API key."
                st.session_state.conversation_history.append((user_input, bot, sa))
                st.session_state.sentiment_history.append({
                    'timestamp': datetime.now(),
                    'message': user_input,
                    'sentiment': sa['sentiment'],
                    'polarity': sa['polarity'],
                    'intensity': sa['intensity'],
                    'urgency_level': sa['urgency_level'],
                    'emotions': sa['emotions'],
                    'confidence': sa['confidence'],
                })
            st.rerun()


with col2:
    st.subheader("📊 Analytics")
    if st.session_state.sentiment_history:
        latest = st.session_state.sentiment_history[-1]
        st.metric("Sentiment", latest['sentiment'].title())
        st.metric("Intensity", latest['intensity'].title())
        st.metric("Urgency", latest['urgency_level'].title())

        sentiments = [x['sentiment'] for x in st.session_state.sentiment_history]
        s_counts = pd.Series(sentiments).value_counts()
        fig_pie = px.pie(values=s_counts.values, names=s_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        priorities = [x['urgency_level'] for x in st.session_state.sentiment_history]
        p_counts = pd.Series(priorities).value_counts()
        fig_priority = px.bar(x=p_counts.index, y=p_counts.values, title="Priority Distribution")
        st.plotly_chart(fig_priority, use_container_width=True)

        if len(st.session_state.sentiment_history) > 1:
            df = pd.DataFrame(st.session_state.sentiment_history)
            fig_line = px.line(df, x='timestamp', y='polarity', title="Sentiment Timeline")
            st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>AI-Powered Chatbot | Streamlit & Gemini AI</div>", unsafe_allow_html=True)