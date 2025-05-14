# 🧠 Acne Severity Detection using MobileNetV2 + GPT Chatbot

A deep learning–powered web app that detects acne severity from facial images and provides personalized skincare advice through a GPT-3.5 chatbot. Built with TensorFlow, Streamlit, and OpenAI's API.

---

## 📌 Overview

This project enables automated acne severity detection using a MobileNetV2 model trained on a balanced image dataset. It integrates an interactive chatbot to assist users in understanding their skin condition and getting tailored skincare advice.

Users upload facial images via a Streamlit web interface, get classified into one of four severity classes—Clear, Mild, Moderate, or Severe—and can ask follow-up skincare questions through OpenAI’s GPT-3.5 model.

---

## ⚙️ Tech Stack

| Layer          | Technology                    |
|----------------|-------------------------------|
| Web Interface  | Streamlit                     |
| Model Inference| TensorFlow + Keras (MobileNetV2) |
| Chatbot        | OpenAI GPT-3.5 API            |
| Image Handling | Pillow                        |
| Environment    | python-dotenv                 |
| Visualization  | Matplotlib                    |

---

## 🌟 Features

- 📷 Upload facial images and receive acne severity classification.
- 🧠 Predicts four classes: `Clear`, `Mild`, `Moderate`, `Severe`.
- ✅ Displays confidence scores with severity-specific recommendations.
- 💬 GPT chatbot responds contextually to acne and skincare queries.
- 🎨 Clean UI with styled headers and background image.
- ⚖️ Handles class imbalance via oversampling and `compute_class_weight`.

---

## 🔁 Project Workflow

1. Dataset is loaded and class-balanced.
2. Images are augmented and preprocessed.
3. MobileNetV2 is trained in two phases:
   - Phase 1: Freeze base model, train custom top layers.
   - Phase 2: Fine-tune last 60 layers with reduced learning rate.
4. Model is saved and used for real-time prediction in Streamlit app.
5. Chatbot answers user questions based on model's predicted class.

---

## 🧩 How to Set Up

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Acne_Severity_Detection.git
cd Acne_Severity_Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv pyenv
# Windows
pyenv\Scripts\activate
# macOS/Linux
source pyenv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Set up the `.env` File

Inside the `dev/` directory, create a `.env` file with:

```env
MODEL_PATH=./models/mobilenetv2_acne_model_improved.keras
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
TRAIN_DIR=./dataset_processed/train
TEST_DIR=./dataset_processed/test
SAMPLE_IMAGE=./dataset_processed/test/acne3_1024/sample.jpg
BACKGROUND_IMAGE=./web_image/bg.png
```

### 5. Run the Web App

```bash
cd dev
streamlit run acne_severity_predictor_app.py
```

---

## 📊 Model Training (Optional)

To train the model from scratch:

```bash
cd dev
python acne_model_trainer.py
```

This will:
- Perform data augmentation
- Handle class imbalance
- Train MobileNetV2 in two stages
- Save model to the `models/` directory

---

## 🚀 Future Enhancements

- Integrate Grad-CAM visualizations for interpretability
- Add optional metadata input (age, gender, skin type)
- Enable offline GPT-based assistant using open-source LLMs
- Deploy to Hugging Face Spaces or Streamlit Cloud
- Extend classification for other skin conditions

---

## 📄 License

MIT License — free to use for educational and research purposes.

---

## 👤 Author

**Nivethini Senthilselvan**  
_M.Sc Applied Machine Intelligence_  
Northeastern University
