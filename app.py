import streamlit as st
import numpy as np
from utils.data_loader import load_pima_dataset
from utils.metrics import *
from models.logistic_model import LogisticRegressionCustom
from models.neural_network import NeuralNetworkCustom

st.set_page_config(page_title="Pima Diabetes - IA", layout="wide")

st.title("Pima Diabetes – Regresión Logística y Red Neuronal")

X, y, X_mean, X_std = load_pima_dataset()

modelo = st.sidebar.selectbox("Modelo", ["Regresión logística", "Red neuronal"])

lr = st.sidebar.number_input("Learning rate", 0.001, 1.0, 0.01)
epochs = st.sidebar.number_input("Iteraciones / Épocas", 500, 10000, 2000)
hidden = st.sidebar.number_input("Neuronas ocultas (RN)", 1, 128, 8)

if st.sidebar.button("Entrenar modelo"):

    # Train-test split
    idx = np.random.permutation(len(X))
    split = int(0.7*len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    if modelo == "Regresión logística":
        model = LogisticRegressionCustom(lr=lr, n_iters=epochs)
    else:
        model = NeuralNetworkCustom(
            n_inputs=X.shape[1],
            n_hidden=hidden,
            lr=lr,
            epochs=epochs
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred)

    st.subheader("Resultados del Modelo")
    st.write(f"Accuracy: {accuracy(y_test, y_pred):.3f}")
    st.write(f"Precisión: {precision(y_test, y_pred):.3f}")
    st.write(f"Sensibilidad (Recall): {recall(y_test, y_pred):.3f}")
    st.write(f"F1-score: {f1_score(y_test, y_pred):.3f}")

    st.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    st.subheader("Matriz de confusión")
    st.write(np.array([[tn, fp],[fn, tp]]))