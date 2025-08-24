# Task 5: Sentiment Analysis Chatbot
import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime

# Initialize Groq client
client = Groq(api_key="YOUR_GROQ_API_KEY")

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Classify sentiment
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Confidence based on absolute polarity
    confidence = min(abs(polarity) * 100, 100)
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'confidence': confidence
    }

def get_sentiment_appropriate_response(user_message, sentiment_data):
    """Generate appropriate response based on sentiment"""
    sentiment = sentiment_data['sentiment']
    polarity = sentiment_data['polarity']
    
    # Create context-aware prompt
    context = f"""
    The user has expressed a {sentiment.lower()} sentiment (polarity: {polarity:.2f}).
    User message: "{user_message}"
    
    Respond appropriately to their emotional state:
    - If positive: Be encouraging and maintain the positive energy
    - If negative: Be empathetic, supportive, and offer help
    - If neutral: Be informative and engaging
    
    Keep the response helpful and emotionally intelligent.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": context}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback responses based on sentiment
        fallback_responses = {
            'Positive': "That's wonderful to hear! I'm glad you're feeling positive. How can I help you today?",
            'Negative': "I understand you might be going through a difficult time. I'm here to help and support you. What can I do for you?",
            'Neutral': "Thank you for your message. I'm here to assist you with whatever you need."
        }
        return fallback_responses.get(sentiment, "I'm here to help you. What would you like to know?")

def create_sentiment_visualization(sentiment_history):
    """Create visualization of sentiment over time"""
    if not sentiment_history:
        return None
    
    df = pd.DataFrame(sentiment_history)
    
    # Create line chart for polarity over time
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['polarity'],
        mode='lines+markers',
        name='Sentiment Polarity',
        line=dict(color='blue'),
        hovertemplate='Time: %{x}<br>Polarity: %{y:.2f}<extra></extra>'
    ))
    
    # Add sentiment zones
    fig.add_hline(y=0.1, line_dash="dash", line_color="green", annotation_text="Positive Zone")
    fig.add_hline(y=-0.1, line_dash="dash", line_color="red", annotation_text="Negative Zone")
    fig.add_hrect(y0=-0.1, y1=0.1, fillcolor="yellow", opacity=0.2, annotation_text="Neutral Zone")
    
    fig.update_layout(
        title="Sentiment Analysis Over Time",
        xaxis_title="Time",
        yaxis_title="Sentiment Polarity",
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

def calculate_satisfaction_metrics(sentiment_history):
    """Calculate customer satisfaction metrics"""
    if not sentiment_history:
        return {}
    
    df = pd.DataFrame(sentiment_history)
    
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()
    total_messages = len(df)
    
    # Calculate metrics
    positive_rate = sentiment_counts.get('Positive', 0) / total_messages * 100
    negative_rate = sentiment_counts.get('Negative', 0) / total_messages * 100
    neutral_rate = sentiment_counts.get('Neutral', 0) / total_messages * 100
    
    # Average sentiment polarity
    avg_polarity = df['polarity'].mean()
    
    # Customer satisfaction score (0-100)
    satisfaction_score = ((avg_polarity + 1) / 2) * 100
    
    return {
        'positive_rate': positive_rate,
        'negative_rate': negative_rate,
        'neutral_rate': neutral_rate,
        'avg_polarity': avg_polarity,
        'satisfaction_score': satisfaction_score,
        'total_interactions': total_messages
    }

def main():
    st.title("💭 Sentiment-Aware Chatbot")
    st.write("A chatbot that understands and responds to your emotions")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sentiment_history' not in st.session_state:
        st.session_state.sentiment_history = []
    
    # Sidebar for analytics
    st.sidebar.title("Sentiment Analytics")
    
    # Display metrics if available
    if st.session_state.sentiment_history:
        metrics = calculate_satisfaction_metrics(st.session_state.sentiment_history)
        
        st.sidebar.metric("Customer Satisfaction", f"{metrics['satisfaction_score']:.1f}%")
        st.sidebar.metric("Total Interactions", metrics['total_interactions'])
        st.sidebar.metric("Positive Rate", f"{metrics['positive_rate']:.1f}%")
        st.sidebar.metric("Negative Rate", f"{metrics['negative_rate']:.1f}%")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # User input
    user_message = st.text_input("Enter your message:", key="user_input")
    
    if st.button("Send") and user_message:
        # Analyze sentiment
        sentiment_data = analyze_sentiment_textblob(user_message)
        
        # Store user message with sentiment
        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
            "sentiment": sentiment_data
        })
        
        # Store sentiment history
        st.session_state.sentiment_history.append({
            'timestamp': datetime.now(),
            'message': user_message,
            'sentiment': sentiment_data['sentiment'],
            'polarity': sentiment_data['polarity'],
            'confidence': sentiment_data['confidence']
        })
        
        # Generate appropriate response
        bot_response = get_sentiment_appropriate_response(user_message, sentiment_data)
        
        # Store bot response
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response
        })
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            sentiment_info = message.get("sentiment", {})
            sentiment_label = sentiment_info.get('sentiment', 'Unknown')
            confidence = sentiment_info.get('confidence', 0)
            
            # Color code based on sentiment
            color_map = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '🟡'}
            emoji = color_map.get(sentiment_label, '⚪')
            
            st.write(f"👤 **You:** {message['content']}")
            st.write(f"{emoji} *Detected: {sentiment_label} ({confidence:.1f}% confidence)*")
        else:
            st.write(f"🤖 **Bot:** {message['content']}")
        st.write("---")
    
    # Sentiment visualization
    if st.session_state.sentiment_history and len(st.session_state.sentiment_history) > 1:
        st.header("📊 Sentiment Analysis")
        
        # Create visualization
        fig = create_sentiment_visualization(st.session_state.sentiment_history)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution pie chart
        df = pd.DataFrame(st.session_state.sentiment_history)
        sentiment_counts = df['sentiment'].value_counts()
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'yellow'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.sentiment_history = []
        st.experimental_rerun()
    
    # Instructions
    st.write("---")
    st.write("**How it works:**")
    st.write("1. Type your message and the bot will analyze your sentiment")
    st.write("2. The bot responds appropriately based on your emotional state")
    st.write("3. View analytics in the sidebar and charts below")

if __name__ == "__main__":
    main()