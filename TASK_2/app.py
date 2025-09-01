# Optimized Multi-modal Chatbot + Image Generator
import os
import io
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from io import BytesIO
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"

client = InferenceClient(
    provider="fal-ai",
    api_key=HUGGINGFACE_API_KEY,
)

def generate_response(prompt, image=None):
    """Generate response using Gemini"""
    try:
        response = gemini_model.generate_content([prompt, image] if image else prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"


def generate_image(prompt):
    """Generate an image using Hugging Face API and save it"""
    try:
        os.makedirs("generated_images", exist_ok=True)
        # output is a PIL.Image object
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-3.5-large",
        )

        return image

    except Exception as e:
        st.error(f"⚠️ Image generation failed: {str(e)}")
        return None


def main():
    st.title("🤖 Multi-modal Chatbot + 🎨 Image Generator")
    st.write("Chat with Gemini AI or generate images with Hugging Face!")

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Image upload for Gemini chat
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    user_image = Image.open(uploaded_file) if uploaded_file else None
    if user_image:
        st.image(user_image, caption="Uploaded Image", use_column_width=True)

    # Tabs for Chat vs Image Generation
    chat_tab, img_tab = st.tabs(["💬 Chat", "🎨 Image Generation"])

    # Chat functionality
    with chat_tab:
        prompt = st.text_input("Enter your message:")
        if st.button("Send") and prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            reply = generate_response(prompt, user_image)
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # Display chat history
        for msg in st.session_state.messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.write(f"{role_icon} **{msg['role'].capitalize()}:** {msg['content']}")

    # Image Generation functionality
    with img_tab:
        img_prompt = st.text_input("Enter a prompt to generate an image:")
        if st.button("Generate Image") and img_prompt:
            gen_img_path = generate_image(img_prompt)
            if gen_img_path:
                st.image(gen_img_path, caption="Generated Image", use_column_width=True)


if __name__ == "__main__":
    main()
