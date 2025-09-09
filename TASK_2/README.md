# Multi-modal Chatbot + Image Generator

This project is an interactive web application that allows users to chat with an AI assistant powered by **Gemini AI** and generate images from text prompts using the **Hugging Face Stable Diffusion API**.
Built using **Streamlit**, it offers a simple interface for conversational interaction and creative image generation.

---

## Overview

The application has two main functionalities:

1. **Chat with Gemini AI**

   * Users can send text messages and optionally upload images.
   * The app uses the Gemini API to generate intelligent responses based on text and uploaded images.

2. **Generate Images from Text Prompts**

   * Users can enter a descriptive text prompt.
   * The app calls the Hugging Face API to generate a corresponding image.

---

## Features

* Multi-modal interaction: Text + Optional Image input for chatbot responses.
* Text-to-Image generation using Stable Diffusion.
* Interactive and easy-to-use Streamlit frontend.
* Persistent chat history during the session.

---

## Prerequisites

* Python 3.8+

* Install dependencies:

  ```bash
  pip install streamlit requests python-dotenv pillow huggingface-hub google-generativeai
  ```

* Create a `.env` file with the following keys:

  ```
  GEMINI_API_KEY=your_gemini_api_key
  HUGGINGFACE_API_KEY=your_huggingface_api_key
  ```

---

## How to Run

1. Run the application:

   ```bash
   streamlit run app.py
   ```

2. Visit `http://localhost:8501` in your web browser.

---

## Application Flow

### Chat Tab

* Type a message and optionally upload an image.
* The app sends the input to the Gemini AI model.
* Displays AI-generated responses in a conversational format.

### Image Generation Tab

* Enter a text prompt.
* The app sends the prompt to Hugging Face’s Stable Diffusion model.
* Displays the generated image in the interface.

---

## Configuration Details

* Gemini Model:
  `"gemini-1.5-flash"`

* Hugging Face Model:
  `"stabilityai/stable-diffusion-3.5-large"`

* Images are saved locally under the folder `generated_images/`.

---

## Notes

* Make sure to manage your API key usage quotas.
* Ensure stable internet connection for API calls.
* Image generation might take a few seconds depending on the prompt complexity.

---

## Possible Improvements

* Add support for conversation context over multiple sessions.
* Implement error handling for API failures.
* Allow users to download generated images.
* Add user authentication for a personalized experience.


---