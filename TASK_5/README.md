
---

# AI-Powered Customer Support Chatbot

This project is a web-based customer support chatbot powered by advanced sentiment and emotion analysis, with the option to integrate Gemini AI for intelligent responses.
It is built using **Streamlit**, providing an easy-to-use interface where users can interact with the chatbot, and support teams can monitor sentiment trends in real time.

---

## Overview

The application enables:

* Real-time conversation with an AI-powered customer support agent.
* Automatic sentiment and emotion analysis of user messages.
* Urgency detection to prioritize critical issues.
* Integration with Gemini AI for advanced responses (optional).
* Interactive analytics dashboard showing sentiment trends over time.

---

## Features

* Analyze sentiment, intensity, urgency, and detected emotions from user messages.
* Fallback responses when no API key is provided.
* Visualizations showing sentiment distribution, priority levels, and sentiment timeline.
* Supports API configuration via a sidebar.
* Clear conversation history with a single click.

---

## Prerequisites

* Python 3.8 or higher
* Install dependencies:

```bash
pip install streamlit pandas numpy textblob plotly google-generativeai
```

* Create a `.env` file with your Gemini API key (if using Gemini):

```
GOOGLE_API_KEY=your_gemini_api_key
```

---

## How to Run

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and go to:
   `http://localhost:8501`

---

## Application Workflow

### Sidebar Configuration

* Select API provider: Gemini AI or Fallback Mode.
* Enter Gemini API key if using Gemini.
* Clear conversation history at any time.

### Chat Interface

* Type your customer message and submit.
* The message is analyzed for sentiment, emotion, intensity, and urgency.
* The AI agent provides a response (via Gemini or fallback logic).
* Conversation history is displayed with sentiment insights.

### Analytics Dashboard

* Displays key metrics from conversation history:

  * Sentiment distribution (Positive, Negative, Neutral).
  * Priority/Urgency distribution.
  * Timeline of sentiment polarity.

---

## Fallback Mode Responses

If no API key is provided, the chatbot uses predefined responses based on the sentiment and urgency of the message.

---

## Future Improvements

* Persistent session management (database integration).
* More advanced multi-modal input (images, attachments).
* User authentication for team usage.
* Downloadable sentiment reports.
* Customizable sentiment and urgency rules.

---


