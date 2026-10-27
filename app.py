
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Prediccion Academica MLP",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Prediccion del Rendimiento Academico")

st.write(
    "Aplicacion web que utiliza una red neuronal MLP "
    "para predecir si un estudiante aprobara o reprobara."
)

st.info(
    "La prediccion es una estimacion del modelo y no garantiza "
    "el resultado academico real."
)


# ==========================================
# CARGAR MODELO Y PREPROCESAMIENTO
# ==========================================

@st.cache_resource
def cargar_recursos():
    modelo = load_model(
        BASE_DIR / "mlp_estudiantes_model.keras"
    )

    imputador = joblib.load(
        BASE_DIR / "imputador_estudiantes.joblib"
    )

    escalador = joblib.load(
        BASE_DIR / "escalador_estudiantes.joblib"
    )

    with open(
        BASE_DIR / "variables_modelo.json",
        "r",
        encoding="utf-8"
    ) as archivo:
        variables = json.load(archivo)

    return modelo, imputador, escalador, variables


archivos_modelo = [
    "mlp_estudiantes_model.keras",
    "imputador_estudiantes.joblib",
    "escalador_estudiantes.joblib",
    "variables_modelo.json"
]

faltantes = [
    nombre
    for nombre in archivos_modelo
    if not (BASE_DIR / nombre).exists()
]

if faltantes:
    st.error(
        "Faltan archivos del modelo. "
        "Verifica los archivos publicados en GitHub."
    )

    for nombre in faltantes:
        st.write(f"- {nombre}")

    st.stop()


try:
    modelo, imputador, escalador, variables = cargar_recursos()

except Exception as error:
    st.error(
        "No se pudieron cargar los archivos del modelo."
    )

    st.exception(error)
    st.stop()


# ==========================================
# FORMULARIO DEL ESTUDIANTE
# ==========================================

st.subheader("Datos del estudiante")

with st.form("formulario_estudiante"):

    col1, col2 = st.columns(2)

    with col1:
        tareas = st.number_input(
            "Tareas (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

        asistencia = st.number_input(
            "Asistencia (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0
        )

        sueno = st.number_input(
            "Sueno (horas por dia)",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

    with col2:
        examenes = st.number_input(
            "Examenes (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0
        )

        estudio = st.number_input(
            "Estudio (horas por semana)",
            min_value=0.0,
            max_value=168.0,
            value=10.0,
            step=1.0
        )

        extracurriculares = st.selectbox(
            "Participa en actividades extracurriculares?",
            options=[1, 0],
            format_func=lambda valor:
                "Si" if valor == 1 else "No"
        )

    enviar = st.form_submit_button(
        "Calcular resultados",
        type="primary",
        use_container_width=True
    )


# ==========================================
# CALCULAR SALIDAS
# ==========================================

if enviar:

    try:
        # Primera salida:
        # Calificacion final calculada con la formula
        # identificada en el dataset.

        calificacion_final = (
            0.3 * tareas
            + 0.4 * examenes
            + 0.1 * asistencia
            + 1.5 * estudio
            + 0.5 * sueno
            + 2.0 * extracurriculares
        )

        # Segunda salida:
        # Prediccion de aprobado mediante la red MLP.

        datos = {
            "Tareas": tareas,
            "Examenes": examenes,
            "Asistencia": asistencia,
            "Estudio_Horas": estudio,
            "Sueno_Horas": sueno,
            "Extracurriculares": extracurriculares
        }

        entrada = pd.DataFrame([datos]).reindex(
            columns=variables
        )

        # Los campos del formulario son obligatorios.
        # Se mantiene el escalador entrenado.

        entrada_escalada = escalador.transform(entrada)

        salida = modelo.predict(
            entrada_escalada,
            verbose=0
        )

        probabilidad = float(
            np.asarray(salida).reshape(-1)[0]
        )

        prediccion = int(probabilidad >= 0.5)

        # Mostrar resultados

        st.divider()
        st.subheader("Resultados del estudiante")

        st.metric(
            "Calificacion final calculada",
            f"{calificacion_final:.2f} / 100"
        )

        if prediccion == 1:
            st.success(
                "Prediccion del modelo: APROBADO"
            )
        else:
            st.error(
                "Prediccion del modelo: REPROBADO"
            )

        st.metric(
            "Probabilidad estimada de aprobar",
            f"{probabilidad * 100:.1f}%"
        )

        st.caption(
            "La calificacion final se calcula mediante la "
            "formula del dataset. La clase aprobado o reprobado "
            "es predicha por la red neuronal MLP."
        )

    except Exception as error:
        st.error(
            "No fue posible calcular los resultados. "
            "Verifica las variables y los archivos del modelo."
        )

        st.exception(error)


# ==========================================
# INFORMACION DEL MODELO
# ==========================================

with st.expander("Acerca del modelo"):

    st.write(
        "Arquitectura: MLP con capas ocultas de "
        "32 y 16 neuronas."
    )

    st.write(
        "Variables de entrada: "
        + ", ".join(variables)
    )

    st.write(
        "Salida de clasificacion: Aprobado "
        "(1 = aprobado, 0 = reprobado)."
    )

    st.write(
        "Calificacion_Final se calcula mediante "
        "una combinacion lineal de las seis variables."
    )
