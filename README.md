

## 🎙️ MOODECHO — Emotion Detection from Speech

**Voice Mood Detector** is a real-time emotion recognition system that analyzes a user’s voice to predict their emotional state. Built using deep learning and audio processing techniques, the app offers a simple web interface where users can either **record their voice live** or **upload a `.wav` file** to detect emotions such as **happy, sad, angry, neutral**, and more.

---

### 🔍 Key Features:

* **Real-time voice recording** using `st_audiorec` in the browser.
* **Automatic emotion detection** using a trained CNN model on MFCC audio features.
* **Intuitive Streamlit UI** for seamless interaction.
* **Confidence scores** for all emotion classes.
* Lightweight model (only \~2MB), ensuring fast and responsive performance.

---

### 🧠 How It Works:

1. **Voice Input**: Users either record or upload a `.wav` audio sample.
2. **Feature Extraction**: The system uses `librosa` to extract MFCC features from the audio.
3. **Model Prediction**: A Convolutional Neural Network (CNN) processes the features and outputs the predicted emotion.
4. **Result Display**: The most likely emotion and associated confidence scores are displayed to the user.

---

### 💡 Tech Stack:

* Python
* Streamlit
* TensorFlow / Keras
* Librosa (audio processing)
* st\_audiorec (voice recording widget)

---

<img width="1832" height="774" alt="Screenshot 2025-08-06 235027" src="https://github.com/user-attachments/assets/c85496fc-d067-4f90-af8a-404ec97401e1" />

