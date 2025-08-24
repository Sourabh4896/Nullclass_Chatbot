# Task 2: Multi-modal Chatbot
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import base64

from dotenv import load_dotenv
import os

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def encode_image(image):
    """Convert PIL image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_response(prompt, image=None):
    """Generate response using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if image:
            response = model.generate_content([prompt, image])
        else:
            response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("🤖 Multi-modal Chatbot")
    st.write("Upload an image and ask questions about it, or just chat!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Chat input
    prompt = st.text_input("Enter your message:")
    
    if st.button("Send") and prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        if uploaded_file:
            response = generate_response(prompt, image)
        else:
            response = generate_response(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"👤 **You:** {message['content']}")
        else:
            st.write(f"🤖 **Bot:** {message['content']}")

if __name__ == "__main__":
    main()