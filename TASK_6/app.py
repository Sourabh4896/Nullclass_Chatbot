# Task 6: Multilingual Chatbot
import streamlit as st
from googletrans import Translator
from groq import Groq

from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# Initialize clients
translator = Translator()
client = Groq(api_key="YOUR_GROQ_API_KEY")

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'hi': 'Hindi',
    'ar': 'Arabic'
}

# Cultural context for responses
CULTURAL_CONTEXTS = {
    'en': {'greeting': 'Hello!', 'politeness': 'formal', 'time_format': '12-hour'},
    'es': {'greeting': '¡Hola!', 'politeness': 'formal', 'time_format': '24-hour'},
    'fr': {'greeting': 'Bonjour!', 'politeness': 'very_formal', 'time_format': '24-hour'},
    'de': {'greeting': 'Hallo!', 'politeness': 'formal', 'time_format': '24-hour'},
    'it': {'greeting': 'Ciao!', 'politeness': 'warm', 'time_format': '24-hour'},
    'pt': {'greeting': 'Olá!', 'politeness': 'warm', 'time_format': '24-hour'},
    'ru': {'greeting': 'Привет!', 'politeness': 'formal', 'time_format': '24-hour'},
    'ja': {'greeting': 'こんにちは!', 'politeness': 'very_formal', 'time_format': '24-hour'},
    'ko': {'greeting': '안녕하세요!', 'politeness': 'very_formal', 'time_format': '12-hour'},
    'zh': {'greeting': '你好!', 'politeness': 'formal', 'time_format': '24-hour'},
    'hi': {'greeting': 'नमस्ते!', 'politeness': 'respectful', 'time_format': '12-hour'},
    'ar': {'greeting': 'مرحبا!', 'politeness': 'formal', 'time_format': '12-hour'}
}

def detect_language(text):
    """Detect language of input text"""
    try:
        detected_lang = detect(text)
        if detected_lang in SUPPORTED_LANGUAGES:
            return detected_lang
        else:
            return 'en'  # Default to English
    except LangDetectException:
        return 'en'  # Default to English if detection fails


def translate_text(text, target_lang, source_lang='auto'):
    """Translate text using Google Translator"""
    try:
        result = translator.translate(text, dest=target_lang, src=source_lang)
        return result.text
    except Exception as e:
        return f"Translation error: {str(e)}"

def get_culturally_appropriate_response(message, detected_lang, target_lang):
    """Generate culturally appropriate response"""
    cultural_context = CULTURAL_CONTEXTS.get(target_lang, CULTURAL_CONTEXTS['en'])
    
    # Translate message to English for processing if needed
    if detected_lang != 'en':
        english_message = translate_text(message, 'en', detected_lang)
    else:
        english_message = message
    
    # Create culturally aware prompt
    prompt = f"""
    Respond to this message in a culturally appropriate way for {SUPPORTED_LANGUAGES[target_lang]} speakers:
    
    Message: "{english_message}"
    
    Cultural context:
    - Greeting style: {cultural_context['greeting']}
    - Politeness level: {cultural_context['politeness']}
    - Be culturally sensitive and appropriate
    
    Respond in English first, then I'll translate it.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        english_response = response.choices[0].message.content
        
        # Translate response to target language if needed
        if target_lang != 'en':
            translated_response = translate_text(english_response, target_lang, 'en')
            return translated_response, english_response
        else:
            return english_response, english_response
            
    except Exception as e:
        fallback = f"I understand you're writing in {SUPPORTED_LANGUAGES[detected_lang]}. How can I help you today?"
        if target_lang != 'en':
            translated_fallback = translate_text(fallback, target_lang, 'en')
            return translated_fallback, fallback
        else:
            return fallback, fallback

def get_language_stats(conversation_history):
    """Get statistics about language usage"""
    if not conversation_history:
        return {}
    
    lang_counts = {}
    for msg in conversation_history:
        lang = msg.get('detected_lang', 'en')
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    total_messages = len(conversation_history)
    lang_percentages = {lang: (count/total_messages)*100 
                       for lang, count in lang_counts.items()}
    
    return {
        'counts': lang_counts,
        'percentages': lang_percentages,
        'total_messages': total_messages,
        'unique_languages': len(lang_counts)
    }

def main():
    st.title("🌍 Multilingual Chatbot")
    st.write("Chat in your preferred language - I'll detect and respond appropriately!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Language selection
    col1, col2 = st.columns(2)
    
    with col1:
        auto_detect = st.checkbox("Auto-detect language", value=True)
    
    with col2:
        if not auto_detect:
            selected_lang = st.selectbox(
                "Choose response language:",
                options=list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]} ({x})"
            )
        else:
            selected_lang = None
    
    # Display supported languages
    with st.expander("Supported Languages"):
        cols = st.columns(3)
        lang_items = list(SUPPORTED_LANGUAGES.items())
        
        for i, (code, name) in enumerate(lang_items):
            col_idx = i % 3
            cols[col_idx].write(f"🌐 {name} ({code})")
    
    # Chat input
    user_message = st.text_input("Enter your message in any supported language:")
    
    if st.button("Send") and user_message:
        # Detect language
        detected_lang = detect_language(user_message)
        
        # Determine target language
        if auto_detect:
            target_lang = detected_lang
        else:
            target_lang = selected_lang
        
        # Add user message to conversation
        user_msg_data = {
            'role': 'user',
            'content': user_message,
            'detected_lang': detected_lang,
            'target_lang': target_lang,
            'original_lang': detected_lang
        }
        
        st.session_state.messages.append(user_msg_data)
        st.session_state.conversation_history.append(user_msg_data)
        
        # Generate response
        response, english_response = get_culturally_appropriate_response(
            user_message, detected_lang, target_lang
        )
        
        # Add bot response
        bot_msg_data = {
            'role': 'assistant',
            'content': response,
            'english_content': english_response,
            'language': target_lang
        }
        
        st.session_state.messages.append(bot_msg_data)
        st.session_state.conversation_history.append(bot_msg_data)
    
    # Display conversation
    st.header("💬 Conversation")
    
    for message in st.session_state.messages:
        if message['role'] == 'user':
            detected = message.get('detected_lang', 'unknown')
            lang_name = SUPPORTED_LANGUAGES.get(detected, 'Unknown')
            
            st.write(f"👤 **You** ({lang_name}):")
            st.write(f"*{message['content']}*")
            
            # Show translation if not in English
            if detected != 'en' and detected in SUPPORTED_LANGUAGES:
                english_version = translate_text(message['content'], 'en', detected)
                st.write(f"🔄 *Translation: {english_version}*")
                
        else:
            lang = message.get('language', 'en')
            lang_name = SUPPORTED_LANGUAGES.get(lang, 'English')
            
            st.write(f"🤖 **Bot** ({lang_name}):")
            st.write(f"*{message['content']}*")
            
            # Show English version if response is in another language
            if lang != 'en':
                st.write(f"🔄 *English: {message.get('english_content', '')}*")
        
        st.write("---")
    
    # Language analytics sidebar
    st.sidebar.header("📊 Language Analytics")
    
    if st.session_state.conversation_history:
        stats = get_language_stats(st.session_state.conversation_history)
        
        st.sidebar.metric("Total Messages", stats['total_messages'])
        st.sidebar.metric("Languages Used", stats['unique_languages'])
        
        st.sidebar.write("**Language Distribution:**")
        for lang_code, percentage in stats['percentages'].items():
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
            st.sidebar.write(f"🌐 {lang_name}: {percentage:.1f}%")
    
    # Quick translation tool
    st.sidebar.header("🔄 Quick Translator")
    
    text_to_translate = st.sidebar.text_input("Text to translate:")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_lang = st.selectbox(
            "From:",
            options=['auto'] + list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: 'Auto-detect' if x == 'auto' else f"{SUPPORTED_LANGUAGES[x]}"
        )
    
    with col2:
        to_lang = st.selectbox(
            "To:",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]}"
        )
    
    if st.sidebar.button("Translate") and text_to_translate:
        translation = translate_text(text_to_translate, to_lang, from_lang)
        st.sidebar.write(f"**Translation:** {translation}")
    
    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.experimental_rerun()
    
    # Instructions
    st.write("---")
    st.write("**Features:**")
    st.write("• Automatic language detection")
    st.write("• Cultural context awareness")
    st.write("• Real-time translation")
    st.write("• Support for 12 languages")
    st.write("• Language usage analytics")

if __name__ == "__main__":
    main()