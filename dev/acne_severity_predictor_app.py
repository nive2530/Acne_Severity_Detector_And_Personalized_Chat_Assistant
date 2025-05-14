# importing the required libraries
import os
from turtle import mode
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import openai
from openai import OpenAI
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_path = os.getenv("MODEL_PATH")
BACKGROUND_IMAGE = os.getenv("BACKGROUND_IMAGE")

# Class labels
class_labels = {
    0: "Clear",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# Recommendations messages
recommendations = {
    "Clear": "👍 Your skin looks clear and healthy!",
    "Mild": "💡 Try a gentle skincare routine with salicylic acid or benzoyl peroxide.",
    "Moderate": "🧴 Consider consulting a dermatologist for topical treatments.",
    "Severe": "⚠️ It's best to consult a dermatologist for a personalized treatment plan."
}

# Load MobileNetV2 model
@st.cache_resource
def load_mobilenetv2_model(model_path):
    return load_model(model_path)

# Predict function
def predict_acne_severity(model_path, uploaded_file, img_size=(160, 160)):
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, height, width, 3)

    # Predict
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = round(preds[0][pred_class] * 100, 2)
    pred_label = class_labels[pred_class]

    print(f"✅ Prediction: {pred_label} ({pred_class}) with confidence: {confidence}%")
    return pred_class,  pred_label, round(confidence,2)

# OpenAI GPT Chat function
def ask_gpt(question, context="I want to know about skincare and acne treatments."):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": question}
    ]
    )
    return response.choices[0].message.content


# Streamlit UI
st.set_page_config(page_title="Predict Your Acne Severity", layout="centered")

def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set local image as background
set_background(BACKGROUND_IMAGE)

st.markdown("""
<div style='
    background-color: #000000;
    padding: 10px;
    border-radius: 10px;
    width: 100%;
    margin: auto;
    text-align: center;
'>
    <h1 style='color: white; margin: 0;'>Predict Your Acne Severity & Ask Me Your Doubts</h1>
</div><br>
""", unsafe_allow_html=True)

st.write("Upload a photo and get instant skin analysis + personalized skincare Q&A.")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_resized = img.resize((150, 200))  # width x height
    st.image(img_resized, caption="Uploaded Image")


    with st.spinner("Analyzing your skin..."):
        model = load_mobilenetv2_model(model_path)
        pred_class, label, confidence = predict_acne_severity(model, uploaded_file)
    
    if label == "Clear" or label == "Mild":
        st.markdown(f"""
        <div style='background-color:#d4edda; padding: 10px; border-radius: 5px;'>
        <strong>Predicted Acne Severity:</strong> <span style='color:#155724'>{label}</span> ({round(confidence,2)}%)
        </div> <br>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#d4edda; padding: 10px; border-radius: 5px;'>
        <strong>Predicted Acne Severity:</strong> <span style='color:#FF0000'>{label}</span> ({confidence}%)
        </div> <br>
        """, unsafe_allow_html=True)
            
    
    
    
    st.markdown(f"""
    <div style='background-color:#e7f3fe; padding: 12px; border-left: 6px solid #2196F3; border-radius: 5px;'>
    <strong> Hey! </strong> {recommendations[label]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("💬 Ask Skincare Questions")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ask new question
    prompt = st.chat_input("Ask a question about acne or skincare...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = ask_gpt(prompt, context=f"My acne severity was detected as {label}. Suggest advice or treatment.")
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
