
---
# AI-Powered Customer Support Chatbot

This project is a web-based customer support chatbot with built-in sentiment and emotion analysis.  
It is built with **Streamlit** and optionally integrates **Gemini AI** for intelligent responses.  

---

## Features

- Real-time AI-powered conversation
- Sentiment and emotion detection (happy, sad, angry, etc.)
- Urgency and intensity analysis of messages
- Visual analytics for support teams:
  - Sentiment distribution
  - Urgency levels
  - Conversation timeline
- Fallback response mode (when no API key is provided)
- Clean UI with conversation history management

---

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install streamlit pandas numpy textblob plotly google-generativeai
````

* (Optional) Add your Gemini API key in a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
```

---

## Run the App

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## How It Works

1. **Sidebar Settings**

   * Choose between Gemini AI or fallback mode
   * Enter Gemini API key if using Gemini
   * Option to clear chat history

2. **Chat Window**

   * Enter a customer query
   * The chatbot analyzes sentiment, detects emotions, and assigns urgency
   * Gemini AI or fallback logic generates a response

3. **Analytics Dashboard**

   * Displays trends in sentiment, urgency, and intensity across conversations

---

## Future Enhancements

* Persistent conversation storage (database)
* Multi-modal inputs (e.g., images, voice)
* Downloadable sentiment reports
* Advanced LLM fine-tuning for domain-specific support

---

