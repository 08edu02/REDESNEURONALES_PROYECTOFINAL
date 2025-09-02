import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import wfdb
import neurokit2 as nk
from tensorflow.keras.models import load_model
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de ECG Local",
    page_icon="❤️",
    layout="wide"
)

# --- Título y Descripción ---
st.title("Análisis y Visualización de Electrocardiogramas (ECG) - Modo Local")
st.markdown("""
Esta aplicación permite visualizar señales de ECG de 12 derivaciones desde **archivos locales**, analizar la frecuencia cardiaca 
y (opcionalmente) clasificar el ritmo cardiaco utilizando un modelo de Deep Learning.
""")

# --- Funciones de Carga y Procesamiento (con caché para optimización) ---

@st.cache_data
def load_record(record_path):
    """
    Carga la señal y metadatos de un registro desde una ruta local.
    Usa el caché de Streamlit para no volver a leer los datos si la ruta no cambia.
    """
    try:
        # Lee directamente desde la ruta local, sin el argumento pn_dir
        record = wfdb.rdrecord(record_path)
        st.success(f"Registro '{record_path}' cargado exitosamente desde la carpeta local.")
        return record
    except Exception as e:
        st.error(f"No se pudo cargar el registro desde '{record_path}'. Verifique que la ruta y los archivos (.hea, .mat) existan. Error: {e}")
        return None

@st.cache_resource
def load_classification_model():
    """
    Carga el modelo de clasificación de Keras pre-entrenado.
    Usa el caché de recursos para que el modelo solo se cargue en memoria una vez.
    """
    model_path = 'ecg_classifier.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

# --- Funciones de Visualización (sin cambios) ---

def plot_ecg_professional(signal_data, metadata):
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    fig, axs = plt.subplots(6, 2, figsize=(20, 25))
    axs = axs.flatten()
    for i, lead_name in enumerate(metadata['sig_name']):
        ax = axs[i]
        signal = signal_data[:, i]
        major_ticks_x = np.arange(0, time[-1], 0.2)
        minor_ticks_x = np.arange(0, time[-1], 0.04)
        major_ticks_y = np.arange(int(np.min(signal)) - 1, int(np.max(signal)) + 1, 0.5)
        minor_ticks_y = np.arange(int(np.min(signal)) - 1, int(np.max(signal)) + 1, 0.1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)
        ax.grid(which='major', linestyle='-', linewidth='0.7', color='red', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='pink', alpha=0.7)
        ax.plot(time, signal, color='black', linewidth=1.2)
        ax.set_title(f"Derivación: {lead_name}")
        ax.set_xlim(0, 10)
        if i % 2 != 0: ax.set_yticklabels([])
        if i < 10: ax.set_xticklabels([])
    fig.supxlabel("Tiempo (s)", fontsize=14)
    fig.supylabel("Voltaje (mV)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def plot_ecg_with_peaks(signal_data, metadata, rpeaks):
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    try:
        lead_index = metadata['sig_name'].index('II')
        lead_name = 'II'
    except ValueError:
        lead_index = 0
        lead_name = metadata['sig_name'][0]
    signal = signal_data[:, lead_index]
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, signal, color='blue', label=f'Señal ECG (Derivación {lead_name})')
    ax.scatter(time[rpeaks], signal[rpeaks], color='red', s=80, marker='x', label='Picos R Detectados')
    ax.set_title("Detección de Picos R para Análisis de Frecuencia Cardiaca", fontsize=16)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Voltaje (mV)")
    ax.legend()
    ax.grid(True, linestyle='--')
    return fig

# --- Barra Lateral (Sidebar) para Controles ---
st.sidebar.header("Panel de Control")
record_path = st.sidebar.text_input(
    "Ingrese la ruta local del registro:",
    "data/WFDBRecords/01/010/JS00001"
)
st.sidebar.markdown("""
**Ejemplos de registros locales:**
- `data/WFDBRecords/01/010/JS00001`
- `data/WFDBRecords/01/010/JS00002`
- `data/WFDBRecords/01/010/JS00004`
""")
st.sidebar.info("Asegúrese de que la estructura de carpetas y los archivos (.hea, .mat) existan.")

# --- Cuerpo Principal de la Aplicación ---
# La variable ahora se llama record_path
record = load_record(record_path)

if record:
    # El resto del código funciona exactamente igual
    signal_mv = record.p_signal
    if record.units[0].lower() == 'uv':
        signal_mv = signal_mv / 1000.0

    with st.expander("Ver Metadatos del Registro", expanded=False):
        st.write(f"**Nombre del Registro:** {record.record_name}")
        st.write(f"**Frecuencia de Muestreo:** {record.fs} Hz")
        # ... (resto de los metadatos)

    st.header("Objetivo 1: Visualización de ECG (12 Derivaciones)")
    fig_professional = plot_ecg_professional(signal_mv, record.__dict__)
    st.pyplot(fig_professional)

    st.header("Objetivo 2: Análisis de Frecuencia Cardiaca")
    try:
        lead_ii_index = record.sig_name.index('II')
    except ValueError:
        lead_ii_index = 0
    ecg_signal_for_analysis = signal_mv[:, lead_ii_index]
    _, rpeaks = nk.ecg_peaks(ecg_signal_for_analysis, sampling_rate=record.fs)
    heart_rate = nk.ecg_rate(rpeaks, sampling_rate=record.fs, desired_length=len(ecg_signal_for_analysis))
    avg_heart_rate = np.mean(heart_rate)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Frecuencia Cardiaca Promedio", value=f"{avg_heart_rate:.2f} lpm")
        if 60 <= avg_heart_rate <= 100:
            st.success("La frecuencia cardiaca está en el rango normal.")
        else:
            st.error("¡Alerta! La frecuencia cardiaca está fuera del rango normal.")
    with col2:
        st.write("Visualización de Picos R:")
        fig_peaks = plot_ecg_with_peaks(signal_mv, record.__dict__, rpeaks['ECG_R_Peaks'])
        st.pyplot(fig_peaks)

    st.header("Objetivo 3 (Opcional): Clasificación de Arritmia con Red Neuronal")
    # ... (el código del objetivo 3 no necesita cambios)
else:
    st.info("Esperando la entrada de una ruta de registro válida en la barra lateral.")