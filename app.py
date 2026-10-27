import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Predicción académica MLP",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Predicción del rendimiento académico")
st.write(
    "Aplicación de demostración que utiliza una red neuronal MLP "
    "para estimar si un estudiante aprobará o reprobará."
)
st.info(
    "La predicción es una estimación del modelo y no garantiza el resultado académico real."
)

@st.cache_resource
def cargar_recursos():
    modelo = load_model(BASE_DIR / "mlp_estudiantes_model.keras")
    imputador = joblib.load(BASE_DIR / "imputador_estudiantes.joblib")
    escalador = joblib.load(BASE_DIR / "escalador_estudiantes.joblib")
    with open(BASE_DIR / "variables_modelo.json", "r", encoding="utf-8") as archivo:
        variables = json.load(archivo)
    return modelo, imputador, escalador, variables

archivos_modelo = [
    "mlp_estudiantes_model.keras",
    "imputador_estudiantes.joblib",
    "escalador_estudiantes.joblib",
    "variables_modelo.json",
]
faltantes = [nombre for nombre in archivos_modelo if not (BASE_DIR / nombre).exists()]

if faltantes:
    st.error("Faltan archivos del modelo. Primero ejecuta el notebook de entrenamiento.")
    st.write("Archivos que faltan:")
    for nombre in faltantes:
        st.write(f"- `{nombre}`")
    st.stop()

try:
    modelo, imputador, escalador, variables = cargar_recursos()
except Exception as error:
    st.error("No se pudieron cargar los archivos del modelo.")
    st.exception(error)
    st.stop()

st.subheader("Datos del estudiante")

with st.form("formulario_estudiante"):
    col1, col2 = st.columns(2)
    with col1:
        tareas = st.number_input("Tareas (0–100)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        asistencia = st.number_input("Asistencia (%)", min_value=0.0, max_value=100.0, value=90.0, step=1.0)
        sueno = st.number_input("Sueño (horas por día)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
    with col2:
        examenes = st.number_input("Exámenes (0–100)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
        estudio = st.number_input("Estudio (horas por semana)", min_value=0.0, max_value=168.0, value=10.0, step=1.0)
        extracurriculares = st.selectbox(
            "¿Participa en actividades extracurriculares?",
            options=[1, 0],
            format_func=lambda valor: "Sí" if valor == 1 else "No",
        )
    enviar = st.form_submit_button("Predecir resultado", type="primary", use_container_width=True)

if enviar:
    datos = {
        "Tareas": tareas,
        "Examenes": examenes,
        "Asistencia": asistencia,
        "Estudio_Horas": estudio,
        "Sueno_Horas": sueno,
        "Extracurriculares": extracurriculares,
    }

    try:
        entrada = pd.DataFrame([datos]).reindex(columns=variables)
        entrada_escalada = escalador.transform(entrada_imputada)
        salida = modelo.predict(entrada_escalada, verbose=0)
        probabilidad = float(np.asarray(salida).reshape(-1)[0])
        prediccion = int(probabilidad >= 0.5)

        st.divider()
        if prediccion == 1:
            st.success("Predicción del modelo: APROBADO")
        else:
            st.error("Predicción del modelo: REPROBADO")
        st.metric("Probabilidad estimada de aprobar", f"{probabilidad * 100:.1f}%")
        st.caption("El resultado depende del modelo entrenado y de los datos suministrados.")
    except Exception as error:
        st.error("No fue posible realizar la predicción. Verifica los archivos y las variables del modelo.")
        st.exception(error)

with st.expander("Acerca del modelo"):
    st.write("Arquitectura: MLP con capas ocultas de 32 y 16 neuronas.")
    st.write("Variables de entrada: " + ", ".join(variables))
    st.write("Variable objetivo: `Aprobado` (1 = aprobado, 0 = reprobado).")
