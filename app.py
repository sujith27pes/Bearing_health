import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import spectrogram, periodogram
from scipy.stats import kurtosis, skew
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import faiss
import os
import ollama
import re

# --- Config ---
sampling_rate = 12000
window_sec = 5
out_shape = (128, 128)
label_map = {0: "Normal", 1: "IR", 2: "OR", 3: "Ball"}
reverse_label_map = {v: k for k, v in label_map.items()}
client = ollama.Client()

# --- Load CNN Model ---
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128, 128, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.load_weights("bearing_fault_cnn_model_new_7.h5")
    return model

# --- Signal Processing ---
def segment_signal(signal, fs, window_sec):
    win_len = fs * window_sec
    return [signal[i:i+win_len] for i in range(0, len(signal), win_len) if len(signal[i:i+win_len]) == win_len]

def segment_to_spectrogram(segment, fs, out_shape):
    f, t, Sxx = spectrogram(segment, fs, nperseg=256, noverlap=128)
    Sxx_db = 10 * np.log10(Sxx + 1e-8)
    h, w = min(Sxx_db.shape[0], out_shape[0]), min(Sxx_db.shape[1], out_shape[1])
    out = np.zeros(out_shape)
    out[:h, :w] = Sxx_db[:h, :w]
    return out

def extract_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    k = kurtosis(signal)
    s = skew(signal)
    p2p = np.ptp(signal)
    crest = np.max(np.abs(signal)) / rms
    f, Pxx = periodogram(signal, sampling_rate)
    dom_freq = f[np.argmax(Pxx)]
    return {
        "rms": round(rms, 4),
        "kurtosis": round(k, 4),
        "skewness": round(s, 4),
        "peak_to_peak": round(p2p, 4),
        "crest_factor": round(crest, 4),
        "dominant_freq": int(dom_freq)
    }

# --- RAG Setup ---
@st.cache_resource
def load_rag_index(kb_path="balanced_bearing_fault_knowledge_base2 (1).txt"):
    with open(kb_path, "r") as f:
        raw_blocks = f.read().strip().split("\n\n")
    texts = [block.replace("\n", " | ") for block in raw_blocks]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index, texts

def retrieve_top_k_cases(features, embedder, index, texts, k=3):
    query_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in features.items()])
    query_embedding = embedder.encode([query_str])
    D, I = index.search(query_embedding, k)
    return [texts[i] for i in I[0]]

def extract_fault_label(text):
    text = text.lower()
    if "inner race" in text or re.search(r"\b(ir|inner\s*race)\b", text): return "IR"
    if "outer race" in text or re.search(r"\b(or|outer\s*race)\b", text): return "OR"
    if "ball" in text: return "Ball"
    if "normal" in text: return "Normal"
    return "Unclear"

def query_llama_local_rag(features, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""
You are a bearing fault diagnosis expert referring to ISO 13373-3 standards.

Use the retrieved cases below as references:

{context_str}

New signal features:
- RMS: {features['rms']}
- Kurtosis: {features['kurtosis']}
- Skewness: {features['skewness']}
- Peak-to-Peak: {features['peak_to_peak']}
- Crest Factor: {features['crest_factor']}
- Dominant Frequency: {features['dominant_freq']} Hz

Return only one word: Normal, IR, OR, or Ball.
"""
    try:
        response = client.generate(model="llama3.2", prompt=prompt)
        return extract_fault_label(response.response.strip())
    except Exception as e:
        return f"LLM Error: {e}"

# --- UI ---
st.set_page_config(page_title="🧠 RAG-Based Bearing Fault Diagnosis", layout="centered")
st.title("🔍 Bearing Fault Diagnosis (CNN + LLM + RAG)")

uploaded_file = st.file_uploader("Upload a .mat vibration file", type=["mat"])
if uploaded_file:
    try:
        mat = loadmat(uploaded_file)
        key = [k for k in mat if "DE_time" in k][0]
        signal = mat[key].ravel()

        segments = segment_signal(signal, sampling_rate, window_sec)
        if not segments:
            st.error("❌ No valid 5-second segment found.")
            st.stop()

        segment = segments[0]
        spec = segment_to_spectrogram(segment, sampling_rate, out_shape)
        features = extract_features(segment)

        # CNN prediction
        model = load_model()
        pred = model.predict(np.expand_dims(spec, axis=(0, -1)))[0]
        cnn_label = label_map[np.argmax(pred)]
        confidence = float(np.max(pred))

        # RAG
        embedder, index, kb_texts = load_rag_index()
        top_contexts = retrieve_top_k_cases(features, embedder, index, kb_texts)
        llm_label = query_llama_local_rag(features, top_contexts)

        # Display
        st.markdown("### 🧪 Extracted Features")
        st.json(features)

        st.markdown("### 📈 CNN Prediction")
        st.success(f"Prediction: **{cnn_label}** with confidence **{confidence:.2%}**")

        st.markdown("### 🤖 LLM (RAG-based) Prediction")
        st.info(f"Prediction: **{llm_label}**")

        st.markdown("### 📊 Segment Preview")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(segment[:2000])
        ax.set_title("First 2000 points of vibration segment")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
