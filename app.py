import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import wfdb
import neurokit2 as nk
from tensorflow.keras.models import load_model
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def plot_ecg_professional_plotly(signal_data, metadata):
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    num_leads = len(metadata['sig_name'])
    
    # ✅ AUMENTAR ESPACIADO VERTICAL Y HORIZONTAL
    fig = make_subplots(
        rows=6, 
        cols=2, 
        subplot_titles=[f"Derivación: {name}" for name in metadata['sig_name']],
        vertical_spacing=0.08,  # Aumentar espacio vertical entre filas
        horizontal_spacing=0.1   # Aumentar espacio horizontal entre columnas
    )
    
    for i, lead_name in enumerate(metadata['sig_name']):
        row = i // 2 + 1
        col = i % 2 + 1
        signal = signal_data[:, i]
        
        fig.add_trace(
            go.Scatter(x=time, y=signal, mode='lines', line=dict(color='black', width=1)),
            row=row, col=col
        )
        
        # Configuración estilo papel ECG
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            dtick=0.2, minor=dict(dtick=0.04, gridcolor='rgba(0,0,0,0.1)', showgrid=True),
            row=row, col=col,
            range=[0, 10]  # Limitar a 10 segundos
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray', 
            dtick=0.5, minor=dict(dtick=0.1, gridcolor='rgba(0,0,0,0.1)', showgrid=True),
            row=row, col=col
        )
    
    # ✅ AUMENTAR ALTURA TOTAL y ajustar márgenes
    fig.update_layout(
        height=1500,  # Aumentar altura
        showlegend=False,
        title_text="ECG - 12 Derivaciones (Estilo Papel Electrocardiográfico)",
        margin=dict(l=50, r=50, t=80, b=50),  # Márgenes ajustados
    )
    
    return fig

# --- Funciones de Visualización (sin cambios) ---

def plot_ecg_professional(signal_data, metadata):
    """
    Crea un gráfico de ECG de 12 derivaciones imitando el papel milimetrado estándar.
    """
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    
    fig, axs = plt.subplots(6, 2, figsize=(20, 25))
    axs = axs.flatten()
    
    for i, lead_name in enumerate(metadata['sig_name']):
        ax = axs[i]
        signal = signal_data[:, i]
        
        # Configuración de la cuadrícula para simular papel de ECG
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
        ax.set_xlim(0, 10) # Los registros son de 10 segundos
        
        # Limpiar etiquetas para una mejor visualización
        if i % 2 != 0: ax.set_yticklabels([])
        if i < 10: ax.set_xticklabels([])

    fig.supxlabel("Tiempo (s)", fontsize=14)
    fig.supylabel("Voltaje (mV)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    return fig

def plot_ecg_with_peaks(signal_data, metadata, rpeaks):
    """
    Grafica una derivación de ECG (usualmente la II) y marca los picos R detectados.
    """
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    
    try:
        lead_index = metadata['sig_name'].index('II')
        lead_name = 'II'
    except ValueError:
        lead_index = 0 # Fallback a la primera derivación si no existe 'II'
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

# ✅ DEFINICIÓN DIRECTA DE REGISTROS
registros = {
    "01/010/JS00001": "data/WFDBRecords/01/010/JS00001",
    "01/010/JS00002": "data/WFDBRecords/01/010/JS00002",
    "01/010/JS00021": "data/WFDBRecords/01/010/JS00021",
    "01/010/JS00043": "data/WFDBRecords/01/010/JS00043",
    "01/010/JS00077": "data/WFDBRecords/01/010/JS00077",
    "02/020/JS01053": "data/WFDBRecords/02/020/JS01053",
    "02/020/JS01072": "data/WFDBRecords/02/020/JS01072",
    "02/020/JS01111": "data/WFDBRecords/02/020/JS01111",
    "02/020/JS01129": "data/WFDBRecords/02/020/JS01129",
    "02/020/JS01156": "data/WFDBRecords/02/020/JS01156"
}

# ✅ SELECTBOX SIMPLE
selected_key = st.sidebar.selectbox(
    "📋 Seleccionar registro:",
    options=list(registros.keys()),
    index=0
)

# ✅ ASIGNACIÓN AUTOMÁTICA
record_path = registros[selected_key]

# --- Cuerpo Principal de la Aplicación ---
record = load_record(record_path)

if record:
    signal_mv = record.p_signal
    if record.units[0].lower() == 'uv':
        signal_mv = signal_mv / 1000.0

    with st.expander("Ver Metadatos del Registro", expanded=False):
        st.write(f"**Nombre del Registro:** {record.record_name}")
        st.write(f"**Frecuencia de Muestreo:** {record.fs} Hz")
        st.write(f"**Duración de la Señal:** {record.sig_len / record.fs} segundos")
        st.write(f"**Número de Derivaciones:** {record.n_sig}")
        st.write(f"**Nombres de Derivaciones:** {', '.join(record.sig_name)}")
        st.write(f"**Unidades:** {', '.join(record.units)}")
        st.write("**Comentarios/Diagnóstico:**")
        for comment in record.comments:
            st.text(f"- {comment}")

    st.header("Objetivo 1: Visualización de ECG (12 Derivaciones)")
    st.markdown("Gráfico de las 12 derivaciones de la señal de ECG, simulando el formato de papel electrocardiográfico estándar (25 mm/s, 10 mm/mV).")
    fig_professional = plot_ecg_professional_plotly(signal_mv, record.__dict__)
    st.plotly_chart(fig_professional, use_container_width=True)

    st.header("Objetivo 2: Análisis de Frecuencia Cardiaca")
    st.markdown("Se utiliza la librería `neurokit2` para detectar los picos R en la derivación II y calcular la frecuencia cardiaca (FC). Se emite una alerta si la FC promedio está fuera del rango normal (60-100 lpm).")
    
    # Seleccionar la derivación II para el análisis
    try:
        lead_ii_index = record.sig_name.index('II')
    except ValueError:
        st.warning("No se encontró la derivación 'II'. Usando la primera derivación para el análisis.")
        lead_ii_index = 0
        
    ecg_signal_for_analysis = signal_mv[:, lead_ii_index]
    
    # Procesar la señal con NeuroKit2
    _, rpeaks = nk.ecg_peaks(ecg_signal_for_analysis, sampling_rate=record.fs)
    heart_rate = nk.ecg_rate(rpeaks, sampling_rate=record.fs, desired_length=len(ecg_signal_for_analysis))
    avg_heart_rate = np.mean(heart_rate)
    
    # Mostrar resultados y alertas
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
    model = load_classification_model()
    if model is None:
        st.warning("""
        **Funcionalidad no disponible.** Para activar la clasificación, debe entrenar un modelo de red neuronal
        y guardarlo como `ecg_classifier.h5` en la misma carpeta que la aplicación.
        """)
    else:
        class_names = ['Sinus Bradycardia', 'Sinus Rhythm', 'Atrial Fibrillation', 'Sinus Tachycardia']
        if st.button("Clasificar Ritmo Cardiaco"):
            with st.spinner("El modelo está analizando la señal..."):
                # Preparar la señal para el modelo (usar la misma derivación y asegurar el tamaño correcto)
                signal_to_classify = ecg_signal_for_analysis
                
                # El modelo espera una forma específica: (batch, timesteps, features)
                signal_reshaped = np.expand_dims(signal_to_classify, axis=0)
                signal_reshaped = np.expand_dims(signal_reshaped, axis=-1)
                
                # Realizar la predicción
                prediction = model.predict(signal_reshaped)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = np.max(prediction) * 100
                
                st.subheader(f"Resultado de la Clasificación: **{predicted_class_name}**")
                st.write(f"Confianza del modelo: **{confidence:.2f}%**")
                
                # Mostrar probabilidades de todas las clases
                probs_df = pd.DataFrame(prediction, columns=class_names, index=["Probabilidad"])
                st.dataframe(probs_df.style.format("{:.2%}"))
else:
    st.info("Esperando la entrada de una ruta de registro válida en la barra lateral.")