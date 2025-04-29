# 🧠 Next Word Prediction using LSTM and GRU RNNs

This project implements a **Next Word Prediction** system using **Recurrent Neural Networks (RNNs)** with both **LSTM** and **GRU** architectures. The model is trained on a given text dataset to predict the next word in a given sequence.

> ⚠️ This project runs **locally** and is built for **learning and experimentation**, so model accuracy may vary.

---

## 🚀 Project Overview

This project:
- Tokenizes a given text dataset.
- Builds n-gram sequences for training.
- Trains an RNN using **LSTM** (or GRU).
- Predicts the next word given an input text sequence.
- Includes a **Streamlit app** for easy interaction.

> 🔗 **[Live Demo](https://predict-next-word-lstm.streamlit.app/)**

---

## 🧰 Technologies Used

- Python 🐍
- TensorFlow / Keras
- NumPy
- Pickle (for saving tokenizer)
- Streamlit (for web UI)

---

## 🛠️ How It Works

1. **Text Preprocessing**
    - Load the text dataset and clean it.
    - Tokenize and convert text to sequences.
    - Create padded n-gram input sequences.

2. **Model Architecture**
    - Embedding Layer
    - LSTM or GRU Layers
    - Dropout for regularization
    - Dense output layer with softmax activation

3. **Training**
    - The model is trained on n-gram sequences.
    - Uses `categorical_crossentropy` loss and `Adam` optimizer.

4. **Prediction**
    - Takes a partial sentence and predicts the next word using the trained model.
