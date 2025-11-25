import streamlit as st
import numpy as np
from utils.data_loader import load_pima_dataset
from utils.metrics import *
from models.logistic_model import LogisticRegressionCustom
from models.neural_network import NeuralNetworkCustom
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pima Diabetes - IA", layout="wide")

# =====================================================
# FUNCIONES PARA IMÁGENES
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
            ax.plot([x_in + 0.03, x_hid - 0.03],
                    [y1, y2], color="gray", linewidth=0.8)

    # Conexiones Oculta → Salida
    for j in range(hidden):
        y2 = 1 - (j + 1) / (hidden + 1)
        ax.plot([x_hid + 0.03, x_out - 0.03],
                [y2, 0.5], color="gray", linewidth=0.8)

    ax.set_title("Arquitectura de la Red Neuronal")
    ax.axis("off")
    plt.tight_layout()
    return fig

# =====================================================
# CARGAR DATOS
# =====================================================

X, y, X_mean, X_std = load_pima_dataset()

# =====================================================
# NAVEGACIÓN
# =====================================================

pagina = st.sidebar.radio(
    "Navegación",
    ["Entrenamiento de Modelos", "Predicción de Paciente", "Créditos"]
)

# =====================================================
# PÁGINA 1 – ENTRENAMIENTO
# =====================================================

if pagina == "Entrenamiento de Modelos":

    st.title("Pima Diabetes – Regresión Logística y Red Neuronal")

    modelo = st.sidebar.selectbox(
        "Modelo", ["Regresión logística", "Red neuronal"]
    )

    lr = st.sidebar.number_input("Learning rate", 0.001, 1.0, 0.05)
    epochs = st.sidebar.number_input("Iteraciones / Épocas", 500, 10000, 5000)
    hidden = st.sidebar.number_input("Neuronas ocultas (RNA)", 1, 128, 8)

    if st.sidebar.button("Entrenar modelo"):

        # Train-test split
        idx = np.random.permutation(len(X))
        split = int(0.7 * len(X))
        X_train, X_test = X[idx[:split]], X[idx[split:]]
        y_train, y_test = y[idx[:split]], y[idx[split:]]
        
        # Crear modelo
        if modelo == "Regresión logística":
            model = LogisticRegressionCustom(lr=lr, n_iters=epochs)
        else:
            model = NeuralNetworkCustom(
                n_inputs=X.shape[1], n_hidden=hidden, lr=lr, epochs=epochs
            )

        # Entrenar
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred)

        # Mostrar métricas
        st.subheader("Resultados del Modelo")
        st.write(f"Accuracy: {accuracy(y_test, y_pred):.3f}")
        st.write(f"Precisión: {precision(y_test, y_pred):.3f}")
        st.write(f"Sensibilidad (Recall): {recall(y_test, y_pred):.3f}")
        st.write(f"F1-score: {f1_score(y_test, y_pred):.3f}")
        st.write(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Mostrar matriz SOLO para regresión logística
        if modelo == "Regresión logística":
            st.subheader("Matriz de Confusión")
            fig_cm = plot_confusion_matrix_image(tn, fp, fn, tp)
            st.pyplot(fig_cm)

        # Mostrar arquitectura SOLO para red neuronal
        if modelo == "Red neuronal":
            st.subheader("Arquitectura de la Red Neuronal")
            fig_arch = plot_architecture(inputs=X.shape[1], hidden=hidden)
            st.pyplot(fig_arch)

# =====================================================
# PÁGINA 2 – PREDICCIÓN DE PACIENTE
# =====================================================

elif pagina == "Predicción de Paciente":

    st.header("Predicción para un nuevo paciente")
    st.write("Ingrese los datos clínicos para obtener una predicción.")

    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
    blood_pressure = st.number_input("BloodPressure", 0.0, 200.0, 70.0)
    skin = st.number_input("SkinThickness", 0.0, 100.0, 20.0)
    insulin = st.number_input("Insulin", 0.0, 600.0, 80.0)
    bmi = st.number_input("BMI", 0.0, 70.0, 30.0)
    pedigree = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predecir paciente"):
        try:
            x_new = np.array([[pregnancies, glucose, blood_pressure, skin,
                               insulin, bmi, pedigree, age]])
            x_norm = (x_new - X_mean) / X_std

            if "model" in globals():
                proba = model.predict_proba(x_norm)[0][0]
                pred = int(proba >= 0.5)

                st.success(f"Probabilidad de diabetes: **{proba:.3f}**")
                st.write(f"Clasificación: **{'DIABETES (1)' if pred == 1 else 'NO DIABETES (0)'}**")
            else:
                st.warning("⚠ Primero entrena un modelo en la sección anterior.")

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# =====================================================
# PÁGINA 3 – CRÉDITOS
# =====================================================

elif pagina == "Créditos":

    st.header("Créditos del Proyecto")
    st.markdown("""
    ### Proyecto académico – Universidad Politécnico Jaime Isaza Cadavid  
    **Curso:** Inteligencia Artificial  
    **Autores:**  
    - Veronica Velez Lotero
    - Isabel Medina
    - Laura Murillo

    ---
    <div style='text-align:center; font-size:13px; color:gray;'>
    © 2025 – Proyecto académico de Inteligencia Artificial.
    </div>
    """, unsafe_allow_html=True)