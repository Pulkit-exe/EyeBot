import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Title of the application
st.title("EyeBot")

# Session state to store the model, messages, and user response
if "messages" not in st.session_state:
    st.session_state.messages = []

if "classification_response" not in st.session_state:
    st.session_state.classification_response = None

# Sidebar for image upload and classification
st.sidebar.header("Have an image? Upload it here....")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load your image classification model
    model = tf.keras.models.load_model("my_model.h5")
    
    # Preprocess the image
    def preprocess_image(image):
        image = image.resize((224, 224))  # Resize the image to the size expected by the model
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    image_array = preprocess_image(image)
    
    # Classify the image
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Map predicted class index to class name
    class_names = [
        "Cataract",
        "Diabetic Retinopathy",
        "Glaucoma",
        "Normal"
    ]
    predicted_class = class_names[predicted_class_index]
    
    st.sidebar.markdown(f"""
        Classification Result: {predicted_class}\n
        Want to know more about this disease?
    """)
    
    # Add Yes/No buttons without page refresh
    if st.sidebar.button("Yes"):
        st.session_state.classification_response = f"Can you tell me more about {predicted_class}?"
    elif st.sidebar.button("No"):
        st.session_state.classification_response = "No further information required."

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response generation
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your custom API to generate the response
    with st.chat_message("assistant"):
        url = 'https://porpoise-on-lark.ngrok-free.app/generate'
        data = {"prompt": prompt}
        response = requests.post(url, json=data)
        result = response.json()
        response_text = result['response']
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Handle classification response if Yes is clicked
if st.session_state.classification_response:
    with st.chat_message("user"):
        st.markdown(st.session_state.classification_response)
    st.session_state.messages.append({"role": "user", "content": st.session_state.classification_response})

    # Call your custom API to generate the response about the disease
    with st.chat_message("assistant"):
        url = 'https://porpoise-on-lark.ngrok-free.app/generate'
        data = {"prompt": st.session_state.classification_response}
        response = requests.post(url, json=data)
        result = response.json()
        response_text = result['response']
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Reset the classification response after handling it
    st.session_state.classification_response = None
