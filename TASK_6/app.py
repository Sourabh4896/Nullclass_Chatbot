import os
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
import re
from google import generativeai as genai

# =====================
# Page configuration
# =====================
st.set_page_config(
    page_title="AI-Powered Multilingual Customer Support Chatbot",
    page_icon="🌐",
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
        "current_language": "auto",
        "detected_languages": [],
        "language_preferences": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =====================
# Language Detection & Translation
# =====================
class MultilingualProcessor:
    def __init__(self):
        self.supported_languages = {
            'en': {
                'name': 'English',
                'flag': '🇺🇸',
                'greeting': 'Hello! How can I help you today?',
                'fallback': 'Thank you for contacting us. I\'m here to help you.',
                'culture': 'direct'
            },
            'es': {
                'name': 'Español',
                'flag': '🇪🇸',
                'greeting': '¡Hola! ¿Cómo puedo ayudarte hoy?',
                'fallback': 'Gracias por contactarnos. Estoy aquí para ayudarte.',
                'culture': 'formal'
            },
            'fr': {
                'name': 'Français',
                'flag': '🇫🇷',
                'greeting': 'Bonjour! Comment puis-je vous aider aujourd\'hui?',
                'fallback': 'Merci de nous avoir contactés. Je suis là pour vous aider.',
                'culture': 'formal'
            },
            'de': {
                'name': 'Deutsch',
                'flag': '🇩🇪',
                'greeting': 'Hallo! Wie kann ich Ihnen heute helfen?',
                'fallback': 'Vielen Dank, dass Sie uns kontaktiert haben. Ich bin hier, um Ihnen zu helfen.',
                'culture': 'very_formal'
            },
            'hi': {
                'name': 'हिन्दी',
                'flag': '🇮🇳',
                'greeting': 'नमस्ते! आज मैं आपकी कैसे सहायता कर सकता हूं?',
                'fallback': 'हमसे संपर्क करने के लिए धन्यवाद। मैं आपकी सहायता के लिए यहां हूं।',
                'culture': 'respectful'
            },
            'ja': {
                'name': '日本語',
                'flag': '🇯🇵',
                'greeting': 'こんにちは！今日はどのようにお手伝いできますか？',
                'fallback': 'お問い合わせいただき、ありがとうございます。お手伝いいたします。',
                'culture': 'very_respectful'
            }
        }
        
        # Language detection patterns
        self.language_patterns = {
            'es': [
                r'\b(hola|buenos días|buenas tardes|gracias|por favor|ayuda|problema)\b',
                r'\b(está|tengo|necesito|quiero|puede|hacer)\b',
                r'ñ', r'¿', r'¡'
            ],
            'fr': [
                r'\b(bonjour|bonsoir|merci|s\'il vous plaît|aide|problème)\b',
                r'\b(je suis|j\'ai|voudrais|pouvez|faire)\b',
                r'\bç\b', r'è|é|ê|ë', r'à|â|ä'
            ],
            'de': [
                r'\b(hallo|guten tag|danke|bitte|hilfe|problem)\b',
                r'\b(ich bin|ich habe|möchte|können|machen)\b',
                r'ß', r'ä|ö|ü'
            ],
            'hi': [
                r'[\u0900-\u097F]+',  # Devanagari script
                r'\b(नमस्ते|धन्यवाद|कृपया|सहायता|समस्या)\b',
                r'\b(मैं|आप|यह|कैसे|क्या)\b'
            ],
            'ja': [
                r'[\u3040-\u309F]+',  # Hiragana
                r'[\u30A0-\u30FF]+',  # Katakana
                r'[\u4E00-\u9FAF]+',  # Kanji
                r'\b(こんにちは|ありがとう|お願い|助け|問題)\b'
            ]
        }
        
        # Cultural response modifiers
        self.cultural_modifiers = {
            'direct': {'formality': 1.0, 'politeness': 1.0, 'elaboration': 1.0},
            'formal': {'formality': 1.3, 'politeness': 1.2, 'elaboration': 1.1},
            'very_formal': {'formality': 1.5, 'politeness': 1.3, 'elaboration': 1.2},
            'respectful': {'formality': 1.2, 'politeness': 1.4, 'elaboration': 1.2},
            'very_respectful': {'formality': 1.4, 'politeness': 1.6, 'elaboration': 1.3}
        }

    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score."""
        if not text or not text.strip():
            return 'en', 0.0
        
        text = text.lower()
        scores = {'en': 0.1}  # Default English baseline
        
        # Pattern-based detection
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 0.3
            
            # Bonus for script detection (especially for Hindi/Japanese)
            if lang in ['hi', 'ja'] and score > 0:
                score *= 2
            
            scores[lang] = score
        
        # TextBlob fallback for additional detection
        try:
            blob = TextBlob(text)
            detected = blob.detect_language()
            if detected in self.supported_languages:
                scores[detected] += 0.5
        except:
            pass
        
        # Find best match
        best_lang = max(scores, key=scores.get)
        confidence = min(scores[best_lang], 1.0)
        
        # Require minimum confidence for non-English
        if best_lang != 'en' and confidence < 0.3:
            return 'en', 0.5
        
        return best_lang, confidence

    def get_language_info(self, lang_code: str) -> Dict:
        """Get language information."""
        return self.supported_languages.get(lang_code, self.supported_languages['en'])

    def get_cultural_context(self, lang_code: str) -> Dict:
        """Get cultural context for response generation."""
        lang_info = self.get_language_info(lang_code)
        culture = lang_info['culture']
        return self.cultural_modifiers.get(culture, self.cultural_modifiers['direct'])

# =====================
# Enhanced Sentiment & Emotion Analyzer
# =====================
class MultilingualSentimentAnalyzer:
    def __init__(self):
        # Multilingual emotion keywords
        self.emotion_keywords = {
            'en': {
                'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid', 'hate'],
                'sad': ['sad', 'depressed', 'unhappy', 'disappointed', 'upset', 'heartbroken', 'miserable', 'terrible'],
                'happy': ['happy', 'excited', 'thrilled', 'delighted', 'pleased', 'satisfied', 'great', 'awesome', 'love'],
                'fearful': ['worried', 'scared', 'afraid', 'anxious', 'concerned', 'nervous', 'panic', 'terrified'],
                'surprised': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow'],
                'confused': ['confused', 'unclear', 'lost', 'puzzled', 'perplexed', "don't understand", 'help'],
            },
            'es': {
                'angry': ['enojado', 'furioso', 'molesto', 'irritado', 'frustrado', 'odio'],
                'sad': ['triste', 'deprimido', 'infeliz', 'decepcionado', 'terrible'],
                'happy': ['feliz', 'emocionado', 'encantado', 'satisfecho', 'genial', 'amor'],
                'fearful': ['preocupado', 'asustado', 'ansioso', 'nervioso', 'pánico'],
                'surprised': ['sorprendido', 'asombrado', 'inesperado'],
                'confused': ['confundido', 'perdido', 'no entiendo', 'ayuda'],
            },
            'fr': {
                'angry': ['en colère', 'furieux', 'énervé', 'irrité', 'frustré', 'haine'],
                'sad': ['triste', 'déprimé', 'malheureux', 'déçu', 'terrible'],
                'happy': ['heureux', 'excité', 'ravi', 'satisfait', 'génial', 'amour'],
                'fearful': ['inquiet', 'effrayé', 'anxieux', 'nerveux', 'panique'],
                'surprised': ['surpris', 'étonné', 'inattendu'],
                'confused': ['confus', 'perdu', 'ne comprends pas', 'aide'],
            },
            'de': {
                'angry': ['wütend', 'sauer', 'verärgert', 'irritiert', 'frustriert', 'hass'],
                'sad': ['traurig', 'deprimiert', 'unglücklich', 'enttäuscht', 'schrecklich'],
                'happy': ['glücklich', 'aufgeregt', 'erfreut', 'zufrieden', 'toll', 'liebe'],
                'fearful': ['besorgt', 'ängstlich', 'nervös', 'panik'],
                'surprised': ['überrascht', 'erstaunt', 'unerwartet'],
                'confused': ['verwirrt', 'verloren', 'verstehe nicht', 'hilfe'],
            },
            'hi': {
                'angry': ['गुस्सा', 'नाराज़', 'क्रोधित', 'परेशान', 'नफरत'],
                'sad': ['दुखी', 'उदास', 'निराश', 'भयानक'],
                'happy': ['खुश', 'प्रसन्न', 'संतुष्ट', 'महान', 'प्यार'],
                'fearful': ['चिंतित', 'डरा हुआ', 'घबराया हुआ'],
                'surprised': ['हैरान', 'अचंभित', 'अप्रत्याशित'],
                'confused': ['भ्रमित', 'समझ नहीं आया', 'मदद'],
            },
            'ja': {
                'angry': ['怒り', '腹立つ', 'イライラ', '嫌い'],
                'sad': ['悲しい', '残念', 'がっかり', 'ひどい'],
                'happy': ['嬉しい', '幸せ', '満足', '素晴らしい', '愛'],
                'fearful': ['心配', '怖い', '不安', 'パニック'],
                'surprised': ['驚き', 'びっくり', '予想外'],
                'confused': ['混乱', 'わからない', '助け'],
            }
        }

        self.intensity_modifiers = {
            'en': {'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'totally': 1.6, 'quite': 1.2, 'somewhat': 0.8},
            'es': {'muy': 1.5, 'realmente': 1.4, 'extremadamente': 1.8, 'totalmente': 1.6, 'bastante': 1.2},
            'fr': {'très': 1.5, 'vraiment': 1.4, 'extrêmement': 1.8, 'totalement': 1.6, 'assez': 1.2},
            'de': {'sehr': 1.5, 'wirklich': 1.4, 'extrem': 1.8, 'völlig': 1.6, 'ziemlich': 1.2},
            'hi': {'बहुत': 1.5, 'वास्तव में': 1.4, 'अत्यधिक': 1.8, 'पूरी तरह': 1.6},
            'ja': {'とても': 1.5, '本当に': 1.4, '非常に': 1.8, '完全に': 1.6}
        }

    def analyze_sentiment(self, text: str, language: str = 'en') -> Dict:
        if not text or not text.strip():
            return {
                'sentiment': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0,
                'confidence': 0.0, 'emotions': {}, 'intensity': 'low', 'urgency_level': 'low',
                'language': language, 'language_confidence': 0.0
            }

        t = text.lower()
        
        # Use TextBlob for basic sentiment (works reasonably for multiple languages)
        try:
            blob = TextBlob(t)
            polarity = float(blob.sentiment.polarity)
            subjectivity = float(blob.sentiment.subjectivity)
        except:
            polarity = 0.0
            subjectivity = 0.5

        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        emotions = self._detect_emotions(t, language)
        intensity = self._calculate_intensity(t, abs(polarity), language)
        urgency_level = self._calculate_urgency(t, sentiment, intensity, emotions, language)
        confidence = min(subjectivity * abs(polarity) * 2.0, 1.0)

        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': confidence,
            'emotions': emotions,
            'intensity': intensity,
            'urgency_level': urgency_level,
            'language': language
        }

    def _detect_emotions(self, text: str, language: str) -> Dict[str, float]:
        emotions: Dict[str, float] = {}
        words = text.split()
        
        # Get emotion keywords for the detected language, fallback to English
        lang_keywords = self.emotion_keywords.get(language, self.emotion_keywords['en'])
        
        for emotion, keywords in lang_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
                    # Check for intensity modifiers
                    if keyword in words:
                        idx = words.index(keyword)
                        if idx > 0:
                            prev = words[idx - 1]
                            modifiers = self.intensity_modifiers.get(language, self.intensity_modifiers['en'])
                            if prev in modifiers:
                                score *= modifiers[prev]
            if score > 0:
                emotions[emotion] = min(score / 3.0, 1.0)
        return emotions

    def _calculate_intensity(self, text: str, base_polarity: float, language: str) -> str:
        intensity_score = base_polarity
        
        # Check for caps ratio
        caps_ratio = (sum(1 for c in text if c.isupper()) / len(text)) if text else 0
        if caps_ratio > 0.3:
            intensity_score *= 1.5
        
        # Check for exclamation marks
        exclam = text.count('!')
        if exclam > 0:
            intensity_score *= (1 + exclam * 0.2)
        
        # Check for repeated punctuation
        if '!!!' in text or '???' in text:
            intensity_score *= 1.3
        
        # Language-specific urgent words
        urgent_words = {
            'en': ['urgent', 'emergency', 'immediately', 'asap', 'critical'],
            'es': ['urgente', 'emergencia', 'inmediatamente', 'crítico'],
            'fr': ['urgent', 'urgence', 'immédiatement', 'critique'],
            'de': ['dringend', 'notfall', 'sofort', 'kritisch'],
            'hi': ['तुरंत', 'आपातकाल', 'जल्दी', 'गंभीर'],
            'ja': ['緊急', '至急', 'すぐに', '重要']
        }
        
        lang_urgent = urgent_words.get(language, urgent_words['en'])
        if any(w in text for w in lang_urgent):
            intensity_score *= 1.4
        
        if intensity_score > 0.7:
            return 'high'
        elif intensity_score > 0.3:
            return 'medium'
        return 'low'

    def _calculate_urgency(self, text: str, sentiment: str, intensity: str, emotions: Dict[str, float], language: str) -> str:
        urgency_score = 0
        
        # Language-specific urgent keywords
        high_urgency = {
            'en': ['emergency', 'urgent', 'critical', 'broken', 'not working', 'error', 'bug', 'refund', 'cancel'],
            'es': ['emergencia', 'urgente', 'crítico', 'roto', 'no funciona', 'error', 'reembolso', 'cancelar'],
            'fr': ['urgence', 'urgent', 'critique', 'cassé', 'ne fonctionne pas', 'erreur', 'remboursement', 'annuler'],
            'de': ['notfall', 'dringend', 'kritisch', 'kaputt', 'funktioniert nicht', 'fehler', 'rückerstattung', 'stornieren'],
            'hi': ['आपातकाल', 'तुरंत', 'गंभीर', 'टूटा', 'काम नहीं कर रहा', 'त्रुटि', 'वापसी', 'रद्द'],
            'ja': ['緊急事態', '緊急', '重要', '壊れた', '動かない', 'エラー', '返金', 'キャンセル']
        }
        
        medium_urgency = {
            'en': ['problem', 'issue', 'help', 'support', 'question', 'confused', 'stuck'],
            'es': ['problema', 'asunto', 'ayuda', 'soporte', 'pregunta', 'confundido'],
            'fr': ['problème', 'question', 'aide', 'support', 'confus'],
            'de': ['problem', 'frage', 'hilfe', 'support', 'verwirrt'],
            'hi': ['समस्या', 'मुद्दा', 'मदद', 'सहायता', 'प्रश्न', 'भ्रमित'],
            'ja': ['問題', 'ヘルプ', 'サポート', '質問', '混乱']
        }
        
        lang_high = high_urgency.get(language, high_urgency['en'])
        lang_medium = medium_urgency.get(language, medium_urgency['en'])
        
        for keyword in lang_high:
            if keyword in text:
                urgency_score += 2
        
        for keyword in lang_medium:
            if keyword in text:
                urgency_score += 1
        
        # Sentiment-based urgency
        if sentiment == 'negative' and intensity == 'high':
            urgency_score += 3
        elif sentiment == 'negative' and intensity == 'medium':
            urgency_score += 2
        
        # Emotion-based urgency
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
# Enhanced AI Response Generator
# =====================
class MultilingualAICustomerSupportAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        self.multilingual_processor = MultilingualProcessor()
        
        try:
            if api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel('gemini-2.0-flash-exp')
        except Exception as e:
            self.client = None
            print(f"Failed to initialize Gemini client: {e}")

        # Enhanced fallback responses by language
        self.fallback_responses = {
            'en': {
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
            },
            'es': {
                'positive': {
                    'high': "¡Me emociona saber que está tan contento! ¿Cómo puedo hacer que su experiencia sea aún mejor?",
                    'medium': "¡Me alegra saber que está satisfecho! ¿Cómo puedo ayudarle más?",
                    'low': "¡Gracias por sus comentarios! ¿Cómo puedo ayudarle hoy?",
                },
                'negative': {
                    'critical': "Pido sinceras disculpas por este problema crítico. Estoy escalando esto inmediatamente a nuestro equipo senior para una resolución urgente.",
                    'high': "Entiendo completamente su frustración, y lo siento mucho. Permítame asegurar personalmente que resolvamos esto para usted inmediatamente.",
                    'medium': "Lamento saber que está teniendo dificultades. Trabajemos juntos para encontrar una solución.",
                    'low': "Me disculpo por las molestias. ¿Con qué problema específico puedo ayudarle?",
                },
                'neutral': {
                    'high': "¡Gracias por contactarnos! Estoy aquí para ayudar con todo lo que necesite.",
                    'medium': "¡Hola! ¿En qué puedo ayudarle hoy?",
                    'low': "¿Cómo puedo asistirle?",
                },
            },
            'fr': {
                'positive': {
                    'high': "Je suis ravi d'apprendre que vous êtes si heureux ! Comment puis-je rendre votre expérience encore meilleure ?",
                    'medium': "Je suis content d'apprendre que vous êtes satisfait ! Comment puis-je vous aider davantage ?",
                    'low': "Merci pour vos commentaires ! Comment puis-je vous aider aujourd'hui ?",
                },
                'negative': {
                    'critical': "Je présente mes sincères excuses pour ce problème critique. J'escalade cela immédiatement à notre équipe senior pour une résolution urgente.",
                    'high': "Je comprends parfaitement votre frustration, et je suis vraiment désolé. Permettez-moi de m'assurer personnellement que nous résolvions cela pour vous immédiatement.",
                    'medium': "Je suis désolé d'apprendre que vous rencontrez des difficultés. Travaillons ensemble pour trouver une solution.",
                    'low': "Je m'excuse pour le dérangement. Avec quel problème spécifique puis-je vous aider ?",
                },
                'neutral': {
                    'high': "Merci de nous avoir contactés ! Je suis ici pour aider avec tout ce dont vous avez besoin.",
                    'medium': "Bonjour ! Que puis-je faire pour vous aider aujourd'hui ?",
                    'low': "Comment puis-je vous assister ?",
                },
            }
        }

    def generate_customer_support_response(self, user_message: str, sentiment_analysis: Dict, detected_language: str) -> str:
        if not self.client:
            return self._get_fallback_response(sentiment_analysis, detected_language)
        
        try:
            prompt = self._create_multilingual_support_prompt(user_message, sentiment_analysis, detected_language)
            response = self.client.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            else:
                return self._get_fallback_response(sentiment_analysis, detected_language)
                
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._get_fallback_response(sentiment_analysis, detected_language)

    def _create_multilingual_support_prompt(self, user_message: str, sa: Dict, detected_language: str) -> str:
        lang_info = self.multilingual_processor.get_language_info(detected_language)
        cultural_context = self.multilingual_processor.get_cultural_context(detected_language)
        
        sentiment = sa['sentiment']
        intensity = sa['intensity']
        emotions = sa['emotions']
        urgency = sa['urgency_level']
        
        emotion_context = ""
        if emotions:
            emotion_list = [f"{k} ({v:.2f})" for k, v in emotions.items() if v > 0.3]
            if emotion_list:
                emotion_context = f"Detected emotions: {', '.join(emotion_list)}. "

        cultural_instructions = self._get_cultural_instructions(cultural_context, lang_info['culture'])

        prompt = f"""
You are an expert multilingual customer support agent. Respond to the following customer message in {lang_info['name']} ({detected_language}):

CUSTOMER MESSAGE: "{user_message}"

LANGUAGE CONTEXT:
- Detected Language: {lang_info['name']} ({lang_info['flag']})
- Cultural Style: {lang_info['culture']}
- {cultural_instructions}

SENTIMENT ANALYSIS:
- Sentiment: {sentiment} ({intensity} intensity)
- Urgency: {urgency}
- {emotion_context}
- Confidence: {sa['confidence']:.2f}

INSTRUCTIONS:
1. Respond ENTIRELY in {lang_info['name']} language
2. Be culturally appropriate for {lang_info['culture']} communication style
3. Be empathetic, professional, and helpful
4. Address the customer's issue directly and provide practical solutions
5. Match your tone to their emotional state and urgency level
6. Use proper grammar and natural expressions in {lang_info['name']}
7. If you cannot help directly, offer alternative solutions or escalation paths
8. Keep response length appropriate to the urgency and complexity

Remember: This is customer service, not translation. Focus on solving their problem while respecting cultural norms.
"""
        
        return prompt

    def _get_cultural_instructions(self, cultural_context: Dict, culture_type: str) -> str:
        formality = cultural_context['formality']
        politeness = cultural_context['politeness']
        elaboration = cultural_context['elaboration']
        
        instructions = []
        
        if culture_type == 'direct':
            instructions.append("Be direct and efficient in your response.")
        elif culture_type == 'formal':
            instructions.append("Use formal language and polite expressions.")
        elif culture_type == 'very_formal':
            instructions.append("Use very formal language, titles, and respectful expressions.")
        elif culture_type == 'respectful':
            instructions.append("Show high respect and use honorific expressions where appropriate.")
        elif culture_type == 'very_respectful':
            instructions.append("Use maximum respect, honorifics, and humble language.")
        
        if formality > 1.2:
            instructions.append("Maintain formal tone throughout.")
        if politeness > 1.3:
            instructions.append("Emphasize politeness and courtesy.")
        if elaboration > 1.2:
            instructions.append("Provide detailed and thorough explanations.")
        
        return " ".join(instructions)

    def _get_fallback_response(self, sa: Dict, detected_language: str) -> str:
        sentiment = sa['sentiment']
        urgency = sa['urgency_level']
        intensity = sa['intensity']
        
        # Get language-specific fallback responses
        lang_responses = self.fallback_responses.get(detected_language, self.fallback_responses['en'])
        
        # Select appropriate response based on urgency/intensity
        key = urgency if urgency in ['critical'] else intensity
        sentiment_responses = lang_responses.get(sentiment, {})
        
        return sentiment_responses.get(key, self.multilingual_processor.get_language_info(detected_language)['fallback'])

# =====================
# Initialize components
# =====================
MULTILINGUAL_PROCESSOR = MultilingualProcessor()
MULTILINGUAL_ANALYZER = MultilingualSentimentAnalyzer()

# =====================
# Sidebar: API config & controls
# =====================
with st.sidebar:
    st.header("🔑 API Configuration")
    st.session_state.api_choice = st.selectbox("Select AI Provider:", ["Gemini AI", "Fallback Mode"], index=0)
    
    resolved_api_key = ""
    if st.session_state.api_choice == "Gemini AI":
        resolved_api_key = st.text_input("Enter Gemini API Key:", type="password")
        if not resolved_api_key:
            resolved_api_key = os.getenv("GOOGLE_API_KEY", "")
        
        if resolved_api_key and (not st.session_state.ai_agent or getattr(st.session_state.ai_agent, 'api_key', None) != resolved_api_key):
            st.session_state.ai_agent = MultilingualAICustomerSupportAgent(api_key=resolved_api_key)
            st.success("✅ Gemini AI connected")
        elif not resolved_api_key:
            st.info("ℹ️ Provide an API key to enable AI responses")
    else:
        st.session_state.ai_agent = MultilingualAICustomerSupportAgent(api_key=None)
        st.info("🔄 Using multilingual fallback mode")

    st.header("🌐 Language Settings")
    
    # Language selection
    language_options = ["auto"] + list(MULTILINGUAL_PROCESSOR.supported_languages.keys())
    language_labels = ["🔄 Auto-detect"] + [
        f"{info['flag']} {info['name']}" 
        for info in MULTILINGUAL_PROCESSOR.supported_languages.values()
    ]
    
    selected_idx = st.selectbox(
        "Preferred Language:",
        range(len(language_options)),
        format_func=lambda i: language_labels[i],
        index=0
    )
    st.session_state.current_language = language_options[selected_idx]
    
    # Display supported languages
    st.subheader("Supported Languages:")
    for code, info in MULTILINGUAL_PROCESSOR.supported_languages.items():
        st.write(f"{info['flag']} {info['name']}")
    
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.sentiment_history = []
        st.session_state.detected_languages = []
        st.rerun()
    
    if st.button("🔄 Reset Language Detection", use_container_width=True):
        st.session_state.detected_languages = []
        st.session_state.language_preferences = {}
        st.rerun()

# =====================
# Main layout
# =====================
st.title("🌐 Multilingual AI Customer Support Chatbot")
st.markdown("*Advanced sentiment analysis with multi-language support and cultural awareness*")

# Language detection status
if st.session_state.detected_languages:
    recent_langs = st.session_state.detected_languages[-3:]  # Show last 3 detections
    lang_display = []
    for lang_code, confidence in recent_langs:
        lang_info = MULTILINGUAL_PROCESSOR.get_language_info(lang_code)
        lang_display.append(f"{lang_info['flag']} {lang_info['name']} ({confidence:.1%})")
    
    st.info(f"🔍 Recently detected: {' → '.join(lang_display)}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")

    chat_container = st.container()
    with chat_container:
        for user_msg, bot_msg, sa, detected_lang in st.session_state.conversation_history:
            lang_info = MULTILINGUAL_PROCESSOR.get_language_info(detected_lang)
            
            st.write(f"**You ({lang_info['flag']} {lang_info['name']}):** {user_msg}")
            
            # Enhanced sentiment display
            emotion_display = []
            if sa.get('emotions'):
                for emotion, score in sa['emotions'].items():
                    if score > 0.3:
                        emotion_display.append(f"{emotion} ({score:.1f})")
            
            emotion_text = f" | Emotions: {', '.join(emotion_display)}" if emotion_display else ""
            
            st.caption(f"Sentiment: {sa['sentiment']} | Intensity: {sa['intensity']} | Urgency: {sa['urgency_level']}{emotion_text}")
            st.write(f"**Bot:** {bot_msg}")
            st.divider()

    # # Example messages in different languages
    # st.subheader("💡 Try These Examples:")
    # example_col1, example_col2, example_col3 = st.columns(3)
    
    # with example_col1:
    #     if st.button("🇺🇸 English", use_container_width=True):
    #         st.session_state.example_input = "I'm having trouble with my order and I'm really frustrated!"
    
    # with example_col2:
    #     if st.button("🇪🇸 Español", use_container_width=True):
    #         st.session_state.example_input = "¡Hola! Tengo un problema urgente con mi facturación."
    
    # with example_col3:
    #     if st.button("🇫🇷 Français", use_container_width=True):
    #         st.session_state.example_input = "Bonjour, j'ai besoin d'aide avec mon compte s'il vous plaît."
    
    # example_col4, example_col5, example_col6 = st.columns(3)
    
    # with example_col4:
    #     if st.button("🇩🇪 Deutsch", use_container_width=True):
    #         st.session_state.example_input = "Ich habe ein dringendes Problem mit meiner Bestellung!"
    
    # with example_col5:
    #     if st.button("🇮🇳 हिन्दी", use_container_width=True):
    #         st.session_state.example_input = "नमस्ते! मुझे अपने खाते के साथ समस्या है।"
    
    # with example_col6:
    #     if st.button("🇯🇵 日本語", use_container_width=True):
    #         st.session_state.example_input = "こんにちは！注文に問題があります。"

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Type your message (in any supported language):",
            value=st.session_state.example_input,
            placeholder="Describe your issue in any language...",
            height=100,
        )
        st.session_state.example_input = ""
        
        submitted = st.form_submit_button("Send Message", use_container_width=True)

        if submitted and user_input.strip():
            with st.spinner("🤖 Analyzing language and generating response..."):
                # Language detection
                if st.session_state.current_language == "auto":
                    detected_lang, lang_confidence = MULTILINGUAL_PROCESSOR.detect_language(user_input)
                else:
                    detected_lang = st.session_state.current_language
                    lang_confidence = 1.0
                
                # Store language detection history
                st.session_state.detected_languages.append((detected_lang, lang_confidence))
                if len(st.session_state.detected_languages) > 10:
                    st.session_state.detected_languages = st.session_state.detected_languages[-10:]
                
                # Sentiment analysis
                sa = MULTILINGUAL_ANALYZER.analyze_sentiment(user_input, detected_lang)
                sa['language_confidence'] = lang_confidence
                
                # Generate response
                if st.session_state.ai_agent:
                    bot_response = st.session_state.ai_agent.generate_customer_support_response(
                        user_input, sa, detected_lang
                    )
                else:
                    lang_info = MULTILINGUAL_PROCESSOR.get_language_info(detected_lang)
                    bot_response = lang_info['greeting']
                
                # Store conversation
                st.session_state.conversation_history.append((user_input, bot_response, sa, detected_lang))
                st.session_state.sentiment_history.append({
                    'timestamp': datetime.now(),
                    'message': user_input,
                    'sentiment': sa['sentiment'],
                    'polarity': sa['polarity'],
                    'intensity': sa['intensity'],
                    'urgency_level': sa['urgency_level'],
                    'emotions': sa['emotions'],
                    'confidence': sa['confidence'],
                    'language': detected_lang,
                    'language_confidence': lang_confidence,
                })
            st.rerun()

with col2:
    st.subheader("📊 Analytics Dashboard")
    
    if st.session_state.sentiment_history:
        latest = st.session_state.sentiment_history[-1]
        lang_info = MULTILINGUAL_PROCESSOR.get_language_info(latest['language'])
        
        # Current status metrics
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Current Language", f"{lang_info['flag']} {lang_info['name']}")
            st.metric("Sentiment", latest['sentiment'].title())
        
        with col_metric2:
            st.metric("Intensity", latest['intensity'].title())
            st.metric("Urgency", latest['urgency_level'].title())
        
        # Language confidence
        st.metric("Language Confidence", f"{latest['language_confidence']:.1%}")
        
        # Sentiment distribution chart
        sentiments = [x['sentiment'] for x in st.session_state.sentiment_history]
        s_counts = pd.Series(sentiments).value_counts()
        fig_pie = px.pie(
            values=s_counts.values, 
            names=s_counts.index, 
            title="Sentiment Distribution",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#808080'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Language usage chart
        languages = [x['language'] for x in st.session_state.sentiment_history]
        lang_counts = pd.Series(languages).value_counts()
        lang_labels = []
        for lang_code in lang_counts.index:
            lang_info = MULTILINGUAL_PROCESSOR.get_language_info(lang_code)
            lang_labels.append(f"{lang_info['flag']} {lang_info['name']}")
        
        fig_lang = px.bar(
            x=lang_labels, 
            y=lang_counts.values, 
            title="Language Usage",
            labels={'x': 'Language', 'y': 'Messages'}
        )
        fig_lang.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lang, use_container_width=True)
        
        # Priority/Urgency distribution
        priorities = [x['urgency_level'] for x in st.session_state.sentiment_history]
        p_counts = pd.Series(priorities).value_counts()
        
        # Custom color mapping for urgency levels
        urgency_colors = {
            'critical': '#8B0000',
            'high': '#DC143C',
            'medium': '#FF8C00',
            'low': '#32CD32'
        }
        
        fig_priority = px.bar(
            x=p_counts.index, 
            y=p_counts.values, 
            title="Urgency Distribution",
            color=p_counts.index,
            color_discrete_map=urgency_colors
        )
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Sentiment timeline (if more than one message)
        if len(st.session_state.sentiment_history) > 1:
            df = pd.DataFrame(st.session_state.sentiment_history)
            
            fig_timeline = px.line(
                df, 
                x='timestamp', 
                y='polarity', 
                title="Sentiment Timeline",
                color='language',
                hover_data=['sentiment', 'intensity', 'urgency_level']
            )
            fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_timeline.update_layout(
                yaxis_title="Sentiment Polarity",
                xaxis_title="Time"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent emotions breakdown
        if latest.get('emotions'):
            emotions_df = pd.DataFrame([
                {'Emotion': emotion.title(), 'Intensity': score}
                for emotion, score in latest['emotions'].items()
                if score > 0.1
            ])
            
            if not emotions_df.empty:
                fig_emotions = px.bar(
                    emotions_df, 
                    x='Emotion', 
                    y='Intensity',
                    title="Current Emotional State",
                    color='Intensity',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_emotions, use_container_width=True)
    
    else:
        st.info("Start a conversation to see analytics!")
        
        # Show language capabilities
        st.subheader("🌍 Language Capabilities")
        for code, info in MULTILINGUAL_PROCESSOR.supported_languages.items():
            with st.expander(f"{info['flag']} {info['name']}", expanded=False):
                st.write(f"**Cultural Style:** {info['culture']}")
                st.write(f"**Sample Greeting:** {info['greeting']}")

# =====================
# Advanced Features Section
# =====================
st.markdown("---")
st.subheader("🚀 Advanced Features")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("""
    **🔍 Automatic Language Detection**
    - Detects 6+ languages automatically
    - Script-based detection for non-Latin scripts
    - Confidence scoring for detection accuracy
    """)

with feature_col2:
    st.markdown("""
    **🌍 Cultural Awareness**
    - Adapts formality levels by culture
    - Respectful communication patterns
    - Language-specific urgency keywords
    """)

with feature_col3:
    st.markdown("""
    **🧠 Advanced Analytics**
    - Multilingual sentiment analysis
    - Cross-language emotion detection
    - Cultural response optimization
    """)

# Footer with technical details
st.markdown("---")
with st.expander("🔧 Technical Details", expanded=False):
    st.markdown("""
    **Language Processing Pipeline:**
    1. **Input Analysis**: Automatic language detection using pattern matching and script analysis
    2. **Sentiment Processing**: Language-specific sentiment analysis with cultural keywords
    3. **Cultural Adaptation**: Response generation adapted to cultural communication norms
    4. **Quality Assurance**: Fallback mechanisms ensure responses in user's preferred language
    
    **Supported Languages:**
    - **English** 🇺🇸: Direct, efficient communication style
    - **Spanish** 🇪🇸: Formal, polite expressions with cultural warmth
    - **French** 🇫🇷: Formal language with politeness emphasis
    - **German** 🇩🇪: Very formal, respectful business communication
    - **Hindi** 🇮🇳: Respectful communication with cultural sensitivity
    - **Japanese** 🇯🇵: Highly respectful, honorific expressions
    
    **Advanced Features:**
    - Script detection for non-Latin alphabets
    - Cultural formality adaptation
    - Multilingual emotion keyword libraries
    - Language confidence scoring
    - Cross-language analytics
    """)

st.markdown(
    "<div style='text-align:center;color:#666;'>🌐 Multilingual AI Chatbot | Powered by Gemini AI & Advanced Language Processing</div>", 
    unsafe_allow_html=True
)