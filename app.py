import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import os
from st_audiorec import st_audiorec

# Load your model
MODEL_PATH = "yess.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion map (adjust based on your model's training)
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Feature extraction from audio (MFCC)
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# Streamlit UI
st.title("🎙️ Real-Time Voice Emotion Detection")
st.write("Record your voice or upload a `.wav` file to detect your emotion.")

# AUDIO RECORDING via st_audiorec
wav_audio_data = st_audiorec()

# Save recorded audio
if wav_audio_data is not None:
    with open("recorded.wav", "wb") as f:
        f.write(wav_audio_data)
    st.audio("recorded.wav")

    features = extract_features("recorded.wav")
    if features is not None:
        features = features[np.newaxis, ..., np.newaxis]  # shape: (1, 40, 174, 1)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_index]

        st.success(f"🧠 Predicted Emotion: **{predicted_emotion.upper()}**")
        st.write("Confidence Scores:")
        for i, label in enumerate(emotion_labels):
            st.write(f"{label}: {prediction[0][i]:.2f}")

        os.remove("recorded.wav")
