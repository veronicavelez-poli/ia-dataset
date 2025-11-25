import streamlit as st
import numpy as np
from utils.data_loader import load_pima_dataset
from utils.metrics import *
from models.logistic_model import LogisticRegressionCustom
from models.neural_network import NeuralNetworkCustom
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pima Diabetes - IA", layout="wide")

st.title("Pima Diabetes – Regresión Logística y Red Neuronal")

# =====================================================
# FUNCIONES PARA STREAMLIT: MATRIZ Y ARQUITECTURA
# =====================================================

def plot_confusion_matrix_image(tn, fp, fn, tp):
    fig, ax = plt.subplots(figsize=(4, 3))
    matriz = [[tn, fp], [fn, tp]]
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    ax.set_xticklabels(["0 (Negativo)", "1 (Positivo)"])
    ax.set_yticklabels(["0 (Negativo)", "1 (Positivo)"])

    return fig


def plot_architecture(inputs, hidden):
    fig, ax = plt.subplots(figsize=(6, 4))

    x_in, x_hid, x_out = 0.1, 0.5, 0.9

    # Neuronas de entrada
    for i in range(inputs):
        ax.scatter(x_in, 1 - (i + 1) / (inputs + 1), 
                   s=500, color="#8bd3dd", edgecolor="black")
        ax.text(x_in - 0.07, 1 - (i + 1) / (inputs + 1), f"E{i+1}", fontsize=8)

    # Neuronas ocultas
    for j in range(hidden):
        ax.scatter(x_hid, 1 - (j + 1) / (hidden + 1), 
                   s=600, color="#ff9f1c", edgecolor="black")
        ax.text(x_hid - 0.03, 1 - (j + 1) / (hidden + 1), f"H{j+1}", fontsize=8)

    # Neurona salida
    ax.scatter(x_out, 0.5, s=700, color="#9ef01a", edgecolor="black")
    ax.text(x_out + 0.03, 0.5, "Salida", fontsize=9)

    # Conexiones Entrada → Oculta
    for i in range(inputs):
        y1 = 1 - (i + 1) / (inputs + 1)
        for j in range(hidden):
            y2 = 1 - (j + 1) / (hidden + 1)
            ax.plot([x_in + 0.03, x_hid - 0.03], [y1, y2], color="gray", linewidth=0.8)

    # Conexiones Oculta → Salida
    for j in range(hidden):
        y2 = 1 - (j + 1) / (hidden + 1)
        ax.plot([x_hid + 0.03, x_out - 0.03], [y2, 0.5], color="gray", linewidth=0.8)

    ax.set_title("Arquitectura de la Red Neuronal")
    ax.axis("off")
    plt.tight_layout()

    return fig

# =====================================================
# DATOS
# =====================================================

X, y, X_mean, X_std = load_pima_dataset()

# =====================================================
# SIDEBAR
# =====================================================

modelo = st.sidebar.selectbox("Modelo", ["Regresión logística", "Red neuronal"])

lr = st.sidebar.number_input("Learning rate", 0.001, 1.0, 0.01)
epochs = st.sidebar.number_input("Iteraciones / Épocas", 500, 10000, 2000)
hidden = st.sidebar.number_input("Neuronas ocultas (RN)", 1, 128, 8)

# =====================================================
# ENTRENAMIENTO
# =====================================================

if st.sidebar.button("Entrenar modelo"):

    # Train-test split
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    # Selección de modelo
    if modelo == "Regresión logística":
        model = LogisticRegressionCustom(lr=lr, n_iters=epochs)
    else:
        model = NeuralNetworkCustom(
            n_inputs=X.shape[1],
            n_hidden=hidden,
            lr=lr,
            epochs=epochs
        )

    # Entrenar
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred)

    # =====================================================
    # RESULTADOS
    # =====================================================
    st.subheader("Resultados del Modelo")
    st.write(f"Accuracy: {accuracy(y_test, y_pred):.3f}")
    st.write(f"Precisión: {precision(y_test, y_pred):.3f}")
    st.write(f"Sensibilidad (Recall): {recall(y_test, y_pred):.3f}")
    st.write(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    st.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # =====================================================
    # MATRIZ DE CONFUSIÓN COMO IMAGEN
    # =====================================================

    st.subheader("Matriz de confusión")
    fig_cm = plot_confusion_matrix_image(tn, fp, fn, tp)
    st.pyplot(fig_cm)

    # =====================================================
    # ARQUITECTURA DE LA RED NEURONAL
    # =====================================================

    if modelo == "Red neuronal":
        st.subheader("Arquitectura de la Red Neuronal")
        fig_arch = plot_architecture(inputs=X.shape[1], hidden=hidden)
        st.pyplot(fig_arch)
    
    # =====================================================
    # PREDICCIÓN DE UN NUEVO PACIENTE
    # =====================================================

    st.subheader("Predicción para un nuevo paciente")

    st.write("Ingresa los valores clínicos del paciente para obtener una predicción usando el modelo entrenado.")

    # Campos para ingresar datos
    col1, col2, col3 = st.columns(3)

    pregnancies = col1.number_input("Pregnancies", 0, 20, 0)
    glucose = col2.number_input("Glucose", 0.0, 300.0, 120.0)
    blood_pressure = col3.number_input("BloodPressure", 0.0, 200.0, 70.0)

    col4, col5, col6 = st.columns(3)

    skin = col4.number_input("SkinThickness", 0.0, 100.0, 20.0)
    insulin = col5.number_input("Insulin", 0.0, 600.0, 80.0)
    bmi = col6.number_input("BMI", 0.0, 70.0, 30.0)

    col7, col8 = st.columns(2)

    pedigree = col7.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
    age = col8.number_input("Age", 1, 120, 30)

    if st.button("Predecir paciente"):
        # Construcción del vector de entrada
        x_new = np.array([[pregnancies, glucose, blood_pressure, skin, insulin, bmi, pedigree, age]])

        # Normalizar igual que los datos de entrenamiento
        x_norm = (x_new - X_mean) / X_std

        # Asegurar que un modelo ya fue entrenado
        if "model" in locals() or "model" in globals():
            proba = model.predict_proba(x_norm)[0][0]
            pred = int(proba >= 0.5)

            st.success(f"Probabilidad de diabetes: **{proba:.3f}**")
            st.write(f"**Clasificación:** {'DIABETES (1)' if pred == 1 else 'NO DIABETES (0)'}")
        else:
            st.warning("Primero entrena un modelo para realizar predicciones.")